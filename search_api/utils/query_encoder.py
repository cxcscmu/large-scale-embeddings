import queue
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer


@dataclass
class EncodingRequest:
    """Container for a single encoding request"""
    request_id: str
    query_text: str
    future: Future
    timestamp: float


class QueryEncoder:
    """
    A batch-processing wrapper for encoding queries using SentenceTransformer models.
    Automatically batches concurrent requests for efficient GPU utilization.
    """

    def __init__(
            self,
            model_name: str = "openbmb/MiniCPM-Embedding-Light",
            use_gpu: Optional[bool] = None,
            use_flash_attention: bool = False,
            batch_size: int = 32,
            batch_timeout_ms: int = 50,
            max_queue_size: int = 1000
    ):
        """
        Initialize the QueryEncoder with batch processing capabilities.

        Args:
            model_name: Name or path of the SentenceTransformer model to load
            use_gpu: Whether to use GPU for encoding. If None, will auto-detect
            use_flash_attention: Whether to use flash attention for faster inference
            batch_size: Maximum number of queries to batch together
            batch_timeout_ms: Maximum time to wait for batch to fill (milliseconds)
            max_queue_size: Maximum number of pending requests in queue
        """
        # Model setup
        if use_gpu is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        # Model configuration
        model_kwargs = {"torch_dtype": torch.float16}
        if use_flash_attention and self.device.type == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using flash attention for faster inference")

        # Load the model
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs=model_kwargs
        )
        self.model.to(self.device)

        # Default instruction/prompt for query encoding
        self.instruction = "Instruction: Given a web search query, retrieve relevant passages that answer the query. Query: "

        # Batch processing configuration
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_queue_size = max_queue_size

        # Request queue and processing thread
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processing_thread.start()

        # Statistics
        self.total_processed = 0
        self.total_batches = 0
        self.stats_lock = threading.Lock()

        print(f"Model '{model_name}' loaded successfully with batch processing")

    def _process_batches(self):
        """Background thread that processes batches of encoding requests"""
        while not self.stop_event.is_set():
            batch = self._collect_batch()
            if batch:
                self._encode_batch(batch)

    def _collect_batch(self) -> List[EncodingRequest]:
        """
        Collect requests for batch processing.
        Returns when batch is full or timeout is reached.
        """
        batch = []
        deadline = time.time() + (self.batch_timeout_ms / 1000.0)

        while len(batch) < self.batch_size:
            try:
                # Calculate remaining time
                timeout = max(0, deadline - time.time())

                if timeout <= 0 and batch:
                    # Timeout reached and we have at least one request
                    break

                # Try to get a request from queue
                request = self.request_queue.get(timeout=timeout if batch else None)
                batch.append(request)

            except queue.Empty:
                if batch:
                    # Timeout reached with partial batch
                    break
                # Continue waiting if no requests yet

        return batch

    def _encode_batch(self, batch: List[EncodingRequest]):
        """Process a batch of encoding requests"""
        try:
            # Extract query texts
            query_texts = [req.query_text for req in batch]
            # print("Encode batch size:", len(query_texts))

            # Batch encode with instruction prefix
            embeddings = self.model.encode(
                query_texts,
                prompt=self.instruction,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # Distribute results to futures
            for i, req in enumerate(batch):
                req.future.set_result(embeddings[i])

            # Update statistics
            with self.stats_lock:
                self.total_processed += len(batch)
                self.total_batches += 1

        except Exception as e:
            # Set exception for all requests in batch
            for req in batch:
                req.future.set_exception(e)

    def encode_query(self, query_text: str, timeout: Optional[float] = 30.0) -> torch.Tensor:
        """
        Encode a single query text into embedding.
        Thread-safe method that can be called concurrently.

        Args:
            query_text: The query text to encode
            timeout: Maximum time to wait for result (seconds)

        Returns:
            A numpy array containing the query embedding
        """
        # Create request with future
        future = Future()
        request = EncodingRequest(
            request_id=str(uuid.uuid4()),
            query_text=query_text,
            future=future,
            timestamp=time.time()
        )

        # Submit to queue
        try:
            self.request_queue.put(request, timeout=1.0)
        except queue.Full:
            raise RuntimeError("Encoding queue is full, too many pending requests")

        # Wait for result
        try:
            embedding = future.result(timeout=timeout)
            return embedding
        except TimeoutError:
            raise TimeoutError(f"Encoding timeout after {timeout} seconds")

    def encode_queries(self, query_texts: List[str], timeout: Optional[float] = 30.0) -> List[torch.Tensor]:
        """
        Encode multiple query texts.
        This method submits all queries at once for efficient batching.

        Args:
            query_texts: List of query texts to encode
            timeout: Maximum time to wait for all results (seconds)

        Returns:
            List of numpy arrays containing query embeddings
        """
        # Create all requests
        requests = []
        for query_text in query_texts:
            future = Future()
            request = EncodingRequest(
                request_id=str(uuid.uuid4()),
                query_text=query_text,
                future=future,
                timestamp=time.time()
            )
            requests.append(request)

        # Submit all to queue
        for request in requests:
            try:
                self.request_queue.put(request, timeout=1.0)
            except queue.Full:
                # Cancel remaining futures
                for req in requests:
                    if not req.future.done():
                        req.future.cancel()
                raise RuntimeError("Encoding queue is full")

        # Collect all results
        results = []
        for request in requests:
            try:
                embedding = request.future.result(timeout=timeout)
                results.append(embedding)
            except Exception as e:
                # Cancel remaining futures on error
                for req in requests:
                    if not req.future.done():
                        req.future.cancel()
                raise e

        return results

    def get_stats(self) -> dict:
        """Get processing statistics"""
        with self.stats_lock:
            avg_batch_size = self.total_processed / self.total_batches if self.total_batches > 0 else 0
            return {
                "total_processed": self.total_processed,
                "total_batches": self.total_batches,
                "average_batch_size": avg_batch_size,
                "queue_size": self.request_queue.qsize(),
                "device": str(self.device)
            }

    def close(self):
        """Gracefully shutdown the encoder"""
        print("Shutting down QueryEncoder...")
        self.stop_event.set()

        # Process remaining requests
        while not self.request_queue.empty():
            time.sleep(0.1)

        self.processing_thread.join(timeout=5.0)
        print("QueryEncoder shutdown complete")


# Example usage and testing
if __name__ == "__main__":
    import concurrent.futures
    import random
    import string


    def generate_random_query(length=50):
        """Generate a random query for testing"""
        words = []
        for _ in range(5, 10):
            word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
            words.append(word)
        return ' '.join(words)


    # Initialize encoder
    encoder = QueryEncoder(
        batch_size=16,  # Process up to 16 queries at once
        batch_timeout_ms=20  # Wait up to 20ms for batch to fill
    )

    print("\n=== Testing Concurrent Encoding ===")

    # Test concurrent encoding
    num_queries = 100
    queries = [generate_random_query() for _ in range(num_queries)]

    start_time = time.time()

    # Submit queries concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for query in queries:
            future = executor.submit(encoder.encode_query, query)
            futures.append(future)

        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                embedding = future.result()
                results.append(embedding)
            except Exception as e:
                print(f"Error: {e}")

    end_time = time.time()

    print(f"\nProcessed {len(results)} queries in {end_time - start_time:.2f} seconds")
    print(f"Average time per query: {(end_time - start_time) / len(results) * 1000:.2f} ms")

    # Print statistics
    stats = encoder.get_stats()
    print("\n=== Processing Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Cleanup
    encoder.close()
