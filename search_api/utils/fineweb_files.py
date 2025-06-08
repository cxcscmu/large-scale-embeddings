import os
import lmdb
import pickle
import time
from typing import Optional, Dict, List, Tuple


class FineWebDocs:
    """
    Class for retrieving document text from sharded LMDB database.
    
    Loads the docid_to_shard mapping and provides efficient access to
    document text across multiple LMDB shards.
    
    Usage:
        doc_manager = FineWebDocs("/path/to/lmdb_shards")
        text = doc_manager.get_txt("document_id")
    """
    
    def __init__(self, base_dir: str, readonly: bool = True, preload_shards: bool = False, 
                max_readers: int = 256, max_cached_envs: int = 10):
        """
        Initialize the FineWebDocs class.
        
        Args:
            base_dir: Base directory containing LMDB shards and docid_to_shard.pkl
            readonly: Whether to open LMDB environments in readonly mode
            preload_shards: Whether to preload all shard environments (memory intensive)
            max_readers: Maximum number of reader slots per environment
            max_cached_envs: Maximum number of environments to keep open if not preloading
        """
        self.base_dir = base_dir
        self.readonly = readonly
        self.max_readers = max_readers
        self.max_cached_envs = max_cached_envs
        
        # Load docid to shard mapping
        self.docid_to_shard = self._load_mapping()
        
        # Get the number of shards from metadata or directory structure
        self.num_shards = self._get_num_shards()
        
        # LMDB environments - either all or cached
        self.envs = {}
        self.transactions = {}
        
        # Usage statistics for LRU cache
        self.env_last_used = {}
        
        # Preload environments if requested
        if preload_shards:
            self._preload_environments()
            
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.access_count = 0
        self.start_time = time.time()
    
    def _load_mapping(self) -> Dict[str, int]:
        """Load the docid_to_shard mapping from pickle file."""
        mapping_path = os.path.join(self.base_dir, "docid_to_shard.pkl")
        
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
        
        print(f"Loading docid_to_shard mapping from {mapping_path}...")
        start_time = time.time()
        
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)
        
        load_time = time.time() - start_time
        print(f"Loaded {len(mapping):,} mappings in {load_time:.2f} seconds")
        
        return mapping
    
    def _get_num_shards(self) -> int:
        """Determine the number of shards from metadata or directory structure."""
        # Try to read from metadata file first
        metadata_path = os.path.join(self.base_dir, "metadata.txt")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                for line in f:
                    if line.startswith("num_shards="):
                        return int(line.strip().split("=")[1])
        
        # Fall back to counting shard directories
        shard_count = 0
        for item in os.listdir(self.base_dir):
            if item.startswith("shard_") and os.path.isdir(os.path.join(self.base_dir, item)):
                shard_count += 1
        
        if shard_count > 0:
            return shard_count
        
        # If all else fails, guess from mapping
        if self.docid_to_shard:
            max_shard = max(self.docid_to_shard.values())
            return max_shard + 1
        
        raise ValueError("Could not determine the number of shards")
    
    def _preload_environments(self):
        """Preload all LMDB environments - memory intensive but faster access."""
        print(f"Preloading {self.num_shards} LMDB environments...")
        for shard_id in range(self.num_shards):
            shard_path = os.path.join(self.base_dir, f"shard_{shard_id:03d}")
            if os.path.isdir(shard_path):
                try:
                    env = lmdb.open(
                        shard_path,
                        readonly=self.readonly,
                        lock=False,
                        readahead=False,
                        max_readers=self.max_readers
                    )
                    self.envs[shard_id] = env
                    # Create and keep a transaction open for faster access
                    self.transactions[shard_id] = env.begin()
                    self.env_last_used[shard_id] = time.time()
                except Exception as e:
                    print(f"Warning: Failed to preload shard {shard_id}: {e}")
        
        print(f"Preloaded {len(self.envs)} environments")
    
    def _get_environment(self, shard_id: int) -> Tuple[lmdb.Environment, lmdb.Transaction]:
        """Get or create LMDB environment and transaction for the specified shard."""
        # Return cached environment if available
        if shard_id in self.envs:
            self.env_last_used[shard_id] = time.time()
            return self.envs[shard_id], self.transactions.get(shard_id)
        
        # Check if we need to close least recently used environments
        if len(self.envs) >= self.max_cached_envs:
            self._close_least_used_env()
        
        # Open the environment
        shard_path = os.path.join(self.base_dir, f"shard_{shard_id:03d}")
        try:
            env = lmdb.open(
                shard_path,
                readonly=self.readonly,
                lock=False,
                readahead=False,
                max_readers=self.max_readers
            )
            
            # Store in cache
            self.envs[shard_id] = env
            txn = env.begin()
            self.transactions[shard_id] = txn
            self.env_last_used[shard_id] = time.time()
            
            return env, txn
        except Exception as e:
            raise RuntimeError(f"Failed to open shard {shard_id}: {e}")
    
    def _close_least_used_env(self):
        """Close the least recently used environment to manage memory usage."""
        if not self.env_last_used:
            return
        
        # Find least recently used
        lru_shard = min(self.env_last_used.items(), key=lambda x: x[1])[0]
        
        # Close the transaction and environment
        if lru_shard in self.transactions:
            self.transactions[lru_shard].abort()
            del self.transactions[lru_shard]
        
        if lru_shard in self.envs:
            self.envs[lru_shard].close()
            del self.envs[lru_shard]
        
        del self.env_last_used[lru_shard]
    
    def get_txt(self, doc_id: str) -> Optional[bytes]:
        """
        Retrieve document text by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document text as bytes or None if not found
        """
        self.access_count += 1
        
        # Look up shard ID from mapping
        if doc_id not in self.docid_to_shard:
            self.miss_count += 1
            return None
        
        shard_id = self.docid_to_shard[doc_id]
        
        try:
            # Get or create environment and transaction
            _, txn = self._get_environment(shard_id)
            
            # Get document bytes
            key = doc_id.encode('utf-8')
            doc_bytes = txn.get(key)
            
            if doc_bytes is not None:
                self.hit_count += 1
                return doc_bytes
            else:
                self.miss_count += 1
                return None
                
        except Exception as e:
            print(f"Error retrieving document {doc_id} from shard {shard_id}: {e}")
            self.miss_count += 1
            return None
    
    def get_multiple_txt(self, doc_ids: List[str]) -> Dict[str, bytes]:
        """
        Efficiently retrieve multiple documents by ID.
        
        Args:
            doc_ids: List of document IDs to retrieve
            
        Returns:
            Dictionary mapping document IDs to their content (bytes)
        """
        # Group by shard for efficiency
        shard_to_ids = {}
        for doc_id in doc_ids:
            self.access_count += 1
            
            if doc_id not in self.docid_to_shard:
                self.miss_count += 1
                continue
            
            shard_id = self.docid_to_shard[doc_id]
            if shard_id not in shard_to_ids:
                shard_to_ids[shard_id] = []
            shard_to_ids[shard_id].append(doc_id)
        
        # Retrieve documents from each shard
        results = {}
        for shard_id, ids in shard_to_ids.items():
            try:
                # Get environment and transaction
                _, txn = self._get_environment(shard_id)
                
                # Retrieve all documents in this shard
                for doc_id in ids:
                    key = doc_id.encode('utf-8')
                    doc_bytes = txn.get(key)
                    
                    if doc_bytes is not None:
                        results[doc_id] = doc_bytes
                        self.hit_count += 1
                    else:
                        self.miss_count += 1
                        
            except Exception as e:
                print(f"Error retrieving documents from shard {shard_id}: {e}")
                self.miss_count += len(ids)
        
        return results
    
    def get_stats(self) -> Dict[str, float]:
        """Get usage statistics for performance monitoring."""
        uptime = time.time() - self.start_time
        
        stats = {
            "access_count": self.access_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / self.access_count if self.access_count > 0 else 0,
            "open_environments": len(self.envs),
            "uptime_seconds": uptime,
            "requests_per_second": self.access_count / uptime if uptime > 0 else 0
        }
        
        return stats
    
    def close(self):
        """Close all open environments and transactions."""
        for shard_id, txn in self.transactions.items():
            txn.abort()
        
        for shard_id, env in self.envs.items():
            env.close()
        
        self.transactions.clear()
        self.envs.clear()
        self.env_last_used.clear()
        
        print("All LMDB environments closed")


# Example usage
if __name__ == "__main__":
    # Initialize document manager
    doc_manager = FineWebDocs(
        base_dir="/bos/tmp2/jening/fineweb_db/lmdb_full",
        readonly=True,
        preload_shards=False,  # Set to True if you have enough memory
        max_cached_envs=10     # Keep 10 most recently used environments open
    )
    
    # Example: Retrieve a single document
    doc_id = "example_doc_id"  # Replace with an actual document ID
    doc_bytes = doc_manager.get_txt(doc_id)
    
    if doc_bytes:
        print(f"Document found: {len(doc_bytes)} bytes")
        # First 100 characters as preview
        try:
            print(f"Preview: {doc_bytes[:100].decode('utf-8', errors='replace')}...")
        except:
            print("Binary content, no preview available")
    else:
        print(f"Document {doc_id} not found")
    
    # Example: Batch retrieval
    doc_ids = ["doc1", "doc2", "doc3"]  # Replace with actual document IDs
    docs = doc_manager.get_multiple_txt(doc_ids)
    print(f"Retrieved {len(docs)} out of {len(doc_ids)} requested documents")
    
    # Print stats
    stats = doc_manager.get_stats()
    print("\nPerformance Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Don't forget to close when done
    doc_manager.close()