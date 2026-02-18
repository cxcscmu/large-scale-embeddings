import os
import re
import lmdb
import threading
import time
import json
from typing import Dict, Optional, Tuple, List

import mmh3  # MurmurHash3 for deterministic shard routing

# Reuse the proven ClueWeb22 filesystem reader (thread-safe, read-only)
# - Provides get_txt(docid) by reading .offset + .json.gz
# - Keeps per-thread LRU of file handles for concurrency safety
from utils.cw22_files import ClueWeb22Docs


class LMDBShardedReader:
    """
    Read-only LMDB shard reader with deterministic routing by hashing the FULL docid.

    - Routing: shard_id = mmh3.hash(docid, signed=False) % num_shards
    - Thread-safe for concurrent reads:
        * Environments are shared across threads with a global lock for first-open
        * Transactions are thread-local (one read txn per shard per thread)
    - No per-document mapping file required
    """

    def __init__(
        self,
        base_dir: str,
        num_shards: Optional[int] = None,
        readonly: bool = True,
        preload_shards: bool = True,
        max_readers: int = 1024,
    ) -> None:
        self.base_dir = base_dir
        self.readonly = readonly
        self.max_readers = max_readers

        # Determine shard count from arg or directory structure
        if num_shards is not None:
            self.num_shards = int(num_shards)
        else:
            # Count shard_* directories
            count = 0
            if os.path.isdir(self.base_dir):
                for item in os.listdir(self.base_dir):
                    p = os.path.join(self.base_dir, item)
                    if item.startswith("shard_") and os.path.isdir(p):
                        count += 1
            if count <= 0:
                raise ValueError(
                    f"No shard_* directories found under {self.base_dir}; pass num_shards explicitly."
                )
            self.num_shards = count

        # Shared environments (across threads); protected by env_lock
        self.envs: Dict[int, lmdb.Environment] = {}
        self.env_lock = threading.Lock()

        # Thread-local read transactions per shard
        self.thread_local = threading.local()

        # Optional warmup: open all envs up front for lower latency
        if preload_shards:
            self._preload_environments()

        # Lightweight stats (best-effort; no strong consistency guarantees)
        self._stats_lock = threading.Lock()
        self._access = 0
        self._hits = 0
        self._miss = 0
        self._t0 = time.time()

    # -------------------------- internals --------------------------
    def _get_thread_transactions(self) -> Dict[int, lmdb.Transaction]:
        if not hasattr(self.thread_local, "transactions"):
            self.thread_local.transactions = {}
        return self.thread_local.transactions

    def _preload_environments(self) -> None:
        with self.env_lock:
            for shard_id in range(self.num_shards):
                if shard_id in self.envs:
                    continue
                shard_path = os.path.join(self.base_dir, f"shard_{shard_id:03d}")
                if not os.path.isdir(shard_path):
                    # Allow sparse/partial sets; the read will error later if missing
                    continue
                env = lmdb.open(
                    shard_path,
                    readonly=self.readonly,
                    lock=False,
                    readahead=False,
                    max_readers=self.max_readers,
                )
                self.envs[shard_id] = env

    def _get_environment(self, shard_id: int) -> Tuple[lmdb.Environment, lmdb.Transaction]:
        # Ensure environment exists (double-checked locking pattern)
        env = self.envs.get(shard_id)
        if env is None:
            with self.env_lock:
                env = self.envs.get(shard_id)
                if env is None:
                    shard_path = os.path.join(self.base_dir, f"shard_{shard_id:03d}")
                    env = lmdb.open(
                        shard_path,
                        readonly=self.readonly,
                        lock=False,
                        readahead=False,
                        max_readers=self.max_readers,
                    )
                    self.envs[shard_id] = env

        # Get thread-local read transaction for this shard (create on first use)
        txns = self._get_thread_transactions()
        txn = txns.get(shard_id)
        if txn is None:
            txn = env.begin()
            txns[shard_id] = txn
        return env, txn

    @staticmethod
    def _shard_for_docid(docid: str, num_shards: int) -> int:
        # Deterministic MurmurHash3 with unsigned 32-bit result for stable modulo behavior
        return mmh3.hash(docid, signed=False) % num_shards

    # --------------------------- public API ---------------------------
    def get_txt(self, docid: str) -> Optional[bytes]:
        """Return the raw JSON bytes for `docid`.

        Assumes LMDB stores the original ClueWeb22 JSON bytes directly
        (no wrapper like {"id": "...", "text": "..."}).
        Returns None if the document is missing or on environment errors.
        """
        with self._stats_lock:
            self._access += 1

        shard_id = self._shard_for_docid(docid, self.num_shards)
        try:
            _, txn = self._get_environment(shard_id)
            val = txn.get(docid.encode("utf-8"))
            if val is None:
                with self._stats_lock:
                    self._miss += 1
                return None

            with self._stats_lock:
                self._hits += 1

            # Directly return the stored bytes
            return val

        except Exception:
            with self._stats_lock:
                self._miss += 1
            return None

    def get_multiple_txt(self, doc_ids: List[str]) -> Dict[str, bytes]:
        # Group docids by shard, then batch get from each txn
        grouping: Dict[int, List[str]] = {}
        for did in doc_ids:
            sid = self._shard_for_docid(did, self.num_shards)
            grouping.setdefault(sid, []).append(did)
        results: Dict[str, bytes] = {}
        for sid, ids in grouping.items():
            try:
                _, txn = self._get_environment(sid)
                for did in ids:
                    val = txn.get(did.encode("utf-8"))
                    with self._stats_lock:
                        self._access += 1
                        if val is None:
                            self._miss += 1
                        else:
                            self._hits += 1
                            results[did] = val
            except Exception:
                with self._stats_lock:
                    self._miss += len(ids)
        return results

    def get_stats(self) -> Dict[str, float]:
        with self._stats_lock:
            acc, hit, miss = self._access, self._hits, self._miss
        dt = max(1e-9, time.time() - self._t0)
        return {
            "num_shards": self.num_shards,
            "open_envs": len(self.envs),
            "access": acc,
            "hits": hit,
            "miss": miss,
            "hit_rate": (hit / acc) if acc else 0.0,
            "qps": acc / dt,
            "uptime_s": dt,
        }

    def close(self) -> None:
        # Abort thread-local read transactions
        if hasattr(self.thread_local, "transactions"):
            for txn in self.thread_local.transactions.values():
                try:
                    txn.abort()
                except Exception:
                    pass
            self.thread_local.transactions.clear()
        # Close environments
        with self.env_lock:
            for env in self.envs.values():
                try:
                    env.close()
                except Exception:
                    pass
            self.envs.clear()


class ClueWeb22DocsTiered:
    """
    Tiered ClueWeb22 reader (read-only, concurrent-safe):
      - If docid belongs to en/en00, read from LMDB shards (hash-based routing)
      - Otherwise, fall back to filesystem reader (ClueWeb22Docs)

    Notes:
      * No per-doc mapping is required for the LMDB path; routing is derived from the full docid.
      * Both backends are read-only and designed for high concurrent access.
    """

    # Reuse the same regex used by ClueWeb22Docs to parse docid components
    _CWID_RE = re.compile(
        r"^clueweb22-((((?:de)|(?:en)|(?:es)|(?:fr)|(?:it)|(?:ja)|(?:nl)|(?:pl)|(?:pt)|(?:zh_chs)|(?:other))(\d{2}))(\d{2}))-(\d{2})-(\d{5})$"
    )

    def __init__(
        self,
        cw22_dir_root: str = "/bos/tmp1/ClueWeb22_L",
        en00_lmdb_dir: str = "/bos/tmp2/jening/clueweb22b_db/lmdb_v2",
        num_shards: Optional[int] = None,
        preload_shards: bool = True,
        max_readers: int = 256,
        max_files_per_thread: int = 10,
        fallback_fs_on_lmdb_miss: bool = False,
    ) -> None:
        # Filesystem reader for non-en00 (and optional fallback for en00 misses)
        self.fs_reader = ClueWeb22Docs(cw22_dir_root=cw22_dir_root, max_files_per_thread=max_files_per_thread)

        # LMDB sharded reader for en00 only
        self.en00_reader = LMDBShardedReader(
            base_dir=en00_lmdb_dir,
            num_shards=num_shards,
            readonly=True,
            preload_shards=preload_shards,
            max_readers=max_readers,
        )

        self.fallback_fs_on_lmdb_miss = fallback_fs_on_lmdb_miss

    @staticmethod
    def _parse(docid: str) -> Tuple[str, str, str, str, str]:
        m = ClueWeb22DocsTiered._CWID_RE.search(docid)
        if not m:
            raise ValueError(f"Invalid ClueWeb22 docid format: {docid}")
        # groups: (subdir, stream, lang, file_seq, doc_seq) interleaved; mirror cw22_files.py semantics
        lang, stream, subdir, file_seq, doc_seq = m.group(3, 2, 1, 6, 7)
        return lang, stream, subdir, file_seq, doc_seq

    # --------------------------- public API ---------------------------
    def get_txt(self, docid: str) -> bytes:
        """Return the decompressed document bytes for the given docid.

        - en/en00 -> from LMDB shards (hash-routed)
        - otherwise -> from filesystem (.offset + .json.gz)
        """
        lang, stream, _, _, _ = self._parse(docid)
        if lang == "en" and stream == "en00":
            val = self.en00_reader.get_txt(docid)
            if val is not None:
                return val
            if self.fallback_fs_on_lmdb_miss:
                # Safety valve: if LMDB is missing a key, optionally fall back to FS
                return self.fs_reader.get_txt(docid)
            raise KeyError(f"Doc not found in LMDB (en00): {docid}")
        # Non-en00: use filesystem path
        return self.fs_reader.get_txt(docid)

    def get_multiple_txt(self, doc_ids: List[str]) -> Dict[str, bytes]:
        """Batch retrieval. Each id is routed individually to the right backend."""
        # Split into en00 and non-en00
        en00_ids: List[str] = []
        fs_ids: List[str] = []
        for did in doc_ids:
            try:
                lang, stream, *_ = self._parse(did)
                if lang == "en" and stream == "en00":
                    en00_ids.append(did)
                else:
                    fs_ids.append(did)
            except Exception:
                # Skip invalid ids silently (or raise if preferred)
                continue

        results: Dict[str, bytes] = {}
        if en00_ids:
            results.update(self.en00_reader.get_multiple_txt(en00_ids))
            if self.fallback_fs_on_lmdb_miss:
                # Identify misses and try filesystem for those
                missed = [did for did in en00_ids if did not in results]
                for did in missed:
                    try:
                        results[did] = self.fs_reader.get_txt(did)
                    except Exception:
                        pass
        # Filesystem docs (non-en00)
        for did in fs_ids:
            try:
                results[did] = self.fs_reader.get_txt(did)
            except Exception:
                pass
        return results

    def get_stats(self) -> Dict[str, float]:
        """Merge lightweight stats from both backends for quick monitoring."""
        stats = {
            "lmdb": self.en00_reader.get_stats(),
            # Filesystem reader does not expose QPS; provide cache size info from current thread
            "fs_thread_cache": self.fs_reader.get_thread_cache_stats(),
        }
        return stats

    def close(self) -> None:
        """Close LMDB environments and release any per-thread filesystem handles."""
        self.en00_reader.close()
        self.fs_reader.cleanup_thread_cache()


if __name__ == "__main__":
    # Tiny smoke test (adjust paths before running)
    tiered = ClueWeb22DocsTiered(
        cw22_dir_root="/bos/tmp1/ClueWeb22_L",
        en00_lmdb_dir="/bos/tmp2/jening/clueweb22b_db/lmdb_full",
        num_shards=32,
        preload_shards=True,
        fallback_fs_on_lmdb_miss=False,
    )

    sample_ids = [
        # en00 ids should be served from LMDB (if present)
        "clueweb22-en0035-22-03042",
        # Non-en00 example (filesystem)
        "clueweb22-en0102-03-00004",
    ]

    for did in sample_ids:
        try:
            b = tiered.get_txt(did)
            print(did, "->", len(b), "bytes")
        except Exception as e:
            print("ERR", did, e)

    print("Stats:", tiered.get_stats())
    tiered.close()
