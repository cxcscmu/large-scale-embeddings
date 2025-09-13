import os
import lmdb
import pickle
import time
import threading
from typing import Optional, Dict, List, Tuple


class FineWebDocs:
    """
    Thread-safe class for retrieving document text from sharded LMDB database.
    
    Loads the docid_to_shard mapping and provides efficient access to
    document text across multiple LMDB shards.
    
    Usage:
        doc_manager = FineWebDocs("/path/to/lmdb_shards")
        text = doc_manager.get_txt("document_id")
    """
    
    def __init__(self, base_dir: str, readonly: bool = True, preload_shards: bool = True, 
                max_readers: int = 256):
        """
        Initialize the FineWebDocs class.
        
        Args:
            base_dir: Base directory containing LMDB shards and docid_to_shard.pkl
            readonly: Whether to open LMDB environments in readonly mode
            preload_shards: Whether to preload all shard environments (recommended as True)
            max_readers: Maximum number of reader slots per environment
        """
        self.base_dir = base_dir
        self.readonly = readonly
        self.max_readers = max_readers
        
        # Load docid to shard mapping
        self.docid_to_shard = self._load_mapping()
        
        # Get the number of shards from metadata or directory structure
        self.num_shards = self._get_num_shards()
        
        # LMDB environments - shared across threads
        self.envs = {}
        self.env_lock = threading.Lock()  # Protect environment operations
        
        # Thread-local storage for transactions
        self.thread_local = threading.local()
        
        # Preload environments (default to True for better performance)
        if preload_shards:
            self._preload_environments()
            
        # Performance tracking (thread-safe)
        self.hit_count = 0
        self.miss_count = 0
        self.access_count = 0
        self.stats_lock = threading.Lock()
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
        """Preload all LMDB environments for better performance."""
        print(f"Preloading {self.num_shards} LMDB environments...")
        with self.env_lock:
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
                    except Exception as e:
                        print(f"Warning: Failed to preload shard {shard_id}: {e}")
        
        print(f"Preloaded {len(self.envs)} environments")
    
    def _get_thread_transactions(self) -> Dict[int, lmdb.Transaction]:
        """Get thread-local transaction dictionary."""
        if not hasattr(self.thread_local, 'transactions'):
            self.thread_local.transactions = {}
        return self.thread_local.transactions
    
    def _get_environment(self, shard_id: int) -> Tuple[lmdb.Environment, lmdb.Transaction]:
        """Get or create LMDB environment and thread-local transaction for the specified shard."""
        # Get or open environment
        with self.env_lock:
            if shard_id not in self.envs:
                # Open the environment on demand
                shard_path = os.path.join(self.base_dir, f"shard_{shard_id:03d}")
                try:
                    env = lmdb.open(
                        shard_path,
                        readonly=self.readonly,
                        lock=False,
                        readahead=False,
                        max_readers=self.max_readers
                    )
                    self.envs[shard_id] = env
                except Exception as e:
                    raise RuntimeError(f"Failed to open shard {shard_id}: {e}")
            
            env = self.envs[shard_id]
        
        # Get or create thread-local transaction
        transactions = self._get_thread_transactions()
        if shard_id not in transactions:
            transactions[shard_id] = env.begin()
        
        return env, transactions[shard_id]
    
    def get_txt(self, doc_id: str) -> Optional[bytes]:
        """
        Retrieve document text by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document text as bytes or None if not found
        """
        with self.stats_lock:
            self.access_count += 1
        
        # Look up shard ID from mapping
        if doc_id not in self.docid_to_shard:
            with self.stats_lock:
                self.miss_count += 1
            return None
        
        shard_id = self.docid_to_shard[doc_id]
        
        try:
            # Get or create environment and thread-local transaction
            _, txn = self._get_environment(shard_id)
            
            # Get document bytes
            key = doc_id.encode('utf-8')
            doc_bytes = txn.get(key)
            
            with self.stats_lock:
                if doc_bytes is not None:
                    self.hit_count += 1
                else:
                    self.miss_count += 1
            
            return doc_bytes
                
        except Exception as e:
            print(f"Error retrieving document {doc_id} from shard {shard_id}: {e}")
            with self.stats_lock:
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
            with self.stats_lock:
                self.access_count += 1
            
            if doc_id not in self.docid_to_shard:
                with self.stats_lock:
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
                # Get environment and thread-local transaction
                _, txn = self._get_environment(shard_id)
                
                # Retrieve all documents in this shard
                for doc_id in ids:
                    key = doc_id.encode('utf-8')
                    doc_bytes = txn.get(key)
                    
                    with self.stats_lock:
                        if doc_bytes is not None:
                            results[doc_id] = doc_bytes
                            self.hit_count += 1
                        else:
                            self.miss_count += 1
                        
            except Exception as e:
                print(f"Error retrieving documents from shard {shard_id}: {e}")
                with self.stats_lock:
                    self.miss_count += len(ids)
        
        return results
    
    def get_stats(self) -> Dict[str, float]:
        """Get usage statistics for performance monitoring."""
        uptime = time.time() - self.start_time
        
        with self.stats_lock:
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
        """Close all open environments and thread-local transactions."""
        # Close any thread-local transactions
        if hasattr(self.thread_local, 'transactions'):
            for txn in self.thread_local.transactions.values():
                txn.abort()
            self.thread_local.transactions.clear()
        
        # Close all environments
        with self.env_lock:
            for env in self.envs.values():
                env.close()
            
            self.envs.clear()
        
        print("All LMDB environments closed")


# Example usage
if __name__ == "__main__":
    # Initialize document manager
    doc_manager = FineWebDocs(
        base_dir="/bos/tmp2/jening/fineweb_db/lmdb_full",
        readonly=True,
        preload_shards=True  # Always preload for best performance
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