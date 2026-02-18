"""
QueryEncoder using infinity_emb for high-throughput, low-latency embedding inference.
Numerical-alignment edition: match the original SentenceTransformer path as closely as possible.
"""

import os
import asyncio
import threading
import time
from typing import List, Optional

import numpy as np
import torch

from concurrent.futures import TimeoutError as FuturesTimeout
from infinity_emb import AsyncEmbeddingEngine, EngineArgs

# ---- Numerical alignment toggles (keep close to original behavior) ----
# 1) Ensure we don't accidentally switch to TF32 math (Infinity acceleration enables it by default in its lib).
try:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends, "cudnn"):
        torch.backends.cuda.matmul.allow_tf32 = False  # match original (default False)
        torch.backends.cudnn.allow_tf32 = False        # match original (default False)
except Exception:
    pass

# 2) Avoid compile-induced tiny numeric drift.
os.environ.setdefault("INFINITY_DISABLE_COMPILE", "TRUE")
# -----------------------------------------------------------------------


class QueryEncoder:
    """
    A high-performance query encoder using infinity_emb backend.
    Synchronous API, thread-safe, and numerically aligned with the original implementation.
    """

    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-Embedding-Light",
        use_gpu: Optional[bool] = None,
        use_flash_attention: bool = False,
        batch_size: int = 32,
        batch_timeout_ms: int = 50,   # kept for compatibility (handled internally)
        max_queue_size: int = 1000,   # kept for compatibility (handled internally)
        # infinity-specific (kept explicit for clarity)
        device: Optional[str] = None,
        engine: str = "torch",
        embedding_dtype: str = "float32",
    ):
        # Device selection (same logic as before)
        if device is None:
            if use_gpu is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size

        print(f"Initializing QueryEncoder with infinity_emb")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")

        # EXACTLY the same instruction string as the original version
        self.instruction = (
            "Instruction: Given a web search query, retrieve relevant passages that answer the query. Query: "
        )

        # We do NOT force flash-attn here (original default is False)
        model_dtype = "float16"  # <- crucial: match original SentenceTransformer torch_dtype=fp16

        # Build engine args with explicit dtype + no BetterTransformer magic
        engine_args = EngineArgs(
            model_name_or_path=model_name,
            engine=engine,                 # torch backend (GPU) to mirror original
            device=device,
            batch_size=batch_size,
            embedding_dtype=embedding_dtype,  # final output dtype
            dtype=model_dtype,                # **compute dtype** aligned to original
            trust_remote_code=True,
            bettertransformer=False,          # keep BT path off for determinism
            model_warmup=False,               # avoid warmup graph variations
        )

        # Single Engine (no list)
        self.engine = AsyncEmbeddingEngine.from_args(engine_args)

        # Stats
        self.total_processed = 0
        self.total_batches = 0
        self._stats_lock = threading.Lock()

        # Async loop in background
        self._loop = None
        self._loop_thread = None
        self._setup_event_loop()

        print(f"Model '{model_name}' loaded successfully with infinity_emb backend")

    def _setup_event_loop(self):
        self._loop_ready = threading.Event()

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop_ready.set()
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        self._loop_ready.wait()

        # Start engine
        future = asyncio.run_coroutine_threadsafe(self.engine.astart(), self._loop)
        future.result(timeout=30)

    def _add_instruction(self, text: str) -> str:
        # Keep exact concatenation (semantically matches SentenceTransformer's prompt=...)
        return self.instruction + text

    def encode_query(self, query_text: str, timeout: Optional[float] = 60.0) -> np.ndarray:
        prefixed_text = self._add_instruction(query_text)

        async def _encode():
            embeddings, _ = await self.engine.embed(sentences=[prefixed_text])
            return embeddings[0]

        future = asyncio.run_coroutine_threadsafe(_encode(), self._loop)
        try:
            emb = future.result(timeout=timeout)
            with self._stats_lock:
                self.total_processed += 1
                self.total_batches += 1
            # Return fp32 arrays (common ST numpy output)
            return np.asarray(emb, dtype=np.float32)
        except FuturesTimeout:
            raise TimeoutError(f"Encoding timeout after {timeout} seconds")

    def encode_queries(self, query_texts: List[str], timeout: Optional[float] = 60.0) -> List[np.ndarray]:
        prefixed_texts = [self._add_instruction(t) for t in query_texts]

        async def _encode():
            embeddings, _ = await self.engine.embed(sentences=prefixed_texts)
            return embeddings

        future = asyncio.run_coroutine_threadsafe(_encode(), self._loop)
        try:
            embs = future.result(timeout=timeout)
            with self._stats_lock:
                self.total_processed += len(query_texts)
                self.total_batches += 1
            return [np.asarray(e, dtype=np.float32) for e in embs]
        except FuturesTimeout:
            raise TimeoutError(f"Encoding timeout after {timeout} seconds")

    def get_stats(self) -> dict:
        with self._stats_lock:
            avg = self.total_processed / self.total_batches if self.total_batches > 0 else 0
            return {
                "total_processed": self.total_processed,
                "total_batches": self.total_batches,
                "average_batch_size": avg,
                "queue_size": 0,
                "device": str(self.device),
            }

    def close(self):
        print("Shutting down QueryEncoder...")
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.engine.astop(), self._loop)
            try:
                future.result(timeout=5.0)
            except Exception as e:
                print(f"Error stopping engine: {e}")
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread:
                self._loop_thread.join(timeout=5.0)
        print("QueryEncoder shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Optional quick check
if __name__ == "__main__":
    enc = QueryEncoder()
    v = enc.encode_query("test")
    print("Shape:", v.shape, "dtype:", v.dtype, "L2 norm:", float(np.linalg.norm(v)))
    enc.close()
