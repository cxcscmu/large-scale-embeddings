import gzip
import os
import re
import threading
from collections import OrderedDict


class ClueWeb22Docs:
    """
    Thread-safe ClueWeb22 document fetcher with thread-local file handle caching.

    Each thread maintains its own LRU cache of file handles (default: 10 files per thread).
    This provides excellent performance for repeated access while keeping FD usage bounded.

    For 20 threads: max FD usage = 20 × 10 = 200 file descriptors.

    Usage:
        cw22_docs = ClueWeb22Docs('/bos/tmp1/ClueWeb22_L')
        doc_text = cw22_docs.get_txt('clueweb22-en0102-03-00004')

        # Optional cleanup (useful in long-running threads)
        cw22_docs.cleanup_thread_cache()
    """

    # Regex pattern for parsing ClueWeb22 document IDs
    cwid_regex = re.compile(
        "^clueweb22-"
        "((((?:de)|(?:en)|(?:es)|(?:fr)|(?:it)|(?:ja)|(?:nl)|(?:pl)|(?:pt)"
        "|(?:zh_chs)|(?:other))"
        "([0-9]{2}))"
        "([0-9]{2}))-"
        "([0-9]{2})-"
        "([0-9]{5})$"
    )

    def __init__(self, cw22_dir_root='/bos/tmp1/ClueWeb22_L', max_files_per_thread=10):
        """
        Initialize ClueWeb22 document fetcher with thread-local file handle caching.

        Args:
            cw22_dir_root: Root directory of ClueWeb22 files (e.g., '/bos/tmp1/ClueWeb22_L')
            max_files_per_thread: Maximum file handles per thread (default: 10)
        """
        # Store path to txt directory (read-only, thread-safe)
        self.cw22_dir_txt = os.path.join(cw22_dir_root, 'txt')

        # Thread-local storage for file handle caching
        self.thread_local = threading.local()
        self.max_files_per_thread = max_files_per_thread

    def __str__(self):
        """Return human-readable information about the instance."""
        return str(self.__dict__)

    def _get_thread_file_cache(self):
        """
        Get or create thread-local file handle cache.

        Returns:
            OrderedDict: LRU cache of file handles for current thread
        """
        if not hasattr(self.thread_local, 'file_cache'):
            # Initialize thread-local LRU cache using OrderedDict
            self.thread_local.file_cache = OrderedDict()
        return self.thread_local.file_cache

    def _get_file_handles(self, lang, stream, subdir, file_seq):
        """
        Get file handles from thread-local cache or open new ones.

        Args:
            lang: Language code
            stream: Stream identifier
            subdir: Subdirectory name
            file_seq: File sequence number

        Returns:
            tuple: (offset_fptr, json_fptr) file handles
        """
        file_cache = self._get_thread_file_cache()
        cache_key = (lang, stream, subdir, file_seq)

        # Check if handles exist in thread-local cache
        if cache_key in file_cache:
            # Move to end (mark as recently used)
            file_handles = file_cache.pop(cache_key)
            file_cache[cache_key] = file_handles
            return file_handles

        # Need to open new file handles
        base_path = os.path.join(self.cw22_dir_txt, lang, stream, subdir,
                                 f'{subdir}-{file_seq}')
        json_path = f'{base_path}.json.gz'
        offset_path = f'{base_path}.offset'

        try:
            # Open new file handles
            offset_fptr = open(offset_path, 'rb')
            json_fptr = open(json_path, 'rb')
            file_handles = (offset_fptr, json_fptr)

            # Check if cache is full and needs eviction
            if len(file_cache) >= self.max_files_per_thread:
                # Remove least recently used item (first item in OrderedDict)
                old_key, old_handles = file_cache.popitem(last=False)
                old_offset_fptr, old_json_fptr = old_handles
                try:
                    old_offset_fptr.close()
                    old_json_fptr.close()
                except:
                    pass  # Ignore errors during cleanup

            # Add new handles to cache
            file_cache[cache_key] = file_handles
            return file_handles

        except (IOError, OSError) as e:
            raise IOError(f"Failed to open files for {cache_key}: {e}")

    def _get_offsets(self, fptr_offset, doc_seq):
        """
        Extract document start and end byte offsets from an open offset file.

        Args:
            fptr_offset: Open file pointer to .offset file
            doc_seq: Document sequence number

        Returns:
            tuple: (doc_start_offset, doc_end_offset)
        """
        offset_entry_length = 11  # 10 digits + newline character

        # Seek to the offset entry for this document
        fptr_offset.seek(int(doc_seq) * offset_entry_length, 0)

        # Read start offset
        doc_start = int(fptr_offset.read(offset_entry_length - 1))
        fptr_offset.read(1)  # Skip newline

        # Read end offset
        doc_end = int(fptr_offset.read(offset_entry_length - 1))

        return (doc_start, doc_end)

    def get_txt(self, docid):
        """
        Get the text representation of a ClueWeb22 document.

        Thread-safe implementation using thread-local file handle caching.
        Each thread maintains its own LRU cache of file handles for efficiency.

        Args:
            docid: ClueWeb22 document ID (e.g., 'clueweb22-en0102-03-00004')

        Returns:
            bytes: Decompressed document content

        Raises:
            ValueError: If docid format is invalid
            IOError: If file reading fails
        """
        # Parse the ClueWeb22 document ID
        matches = self.cwid_regex.search(docid)
        if not matches:
            raise ValueError(f"Invalid ClueWeb22 docid format: {docid}")

        # Extract components from regex groups
        lang, stream, subdir, file_seq, doc_seq = matches.group(3, 2, 1, 6, 7)

        try:
            # Get file handles from thread-local cache
            offset_fptr, json_fptr = self._get_file_handles(lang, stream, subdir, file_seq)

            # Extract document byte offsets
            doc_start, doc_end = self._get_offsets(offset_fptr, doc_seq)

            # Read compressed document data
            json_fptr.seek(doc_start, 0)
            compressed_doc = json_fptr.read(doc_end - doc_start)

            # Decompress and return
            return gzip.decompress(compressed_doc)

        except (IOError, OSError) as e:
            raise IOError(f"Failed to read document {docid}: {e}")

    def cleanup_thread_cache(self):
        """
        Manually cleanup thread-local file handle cache.
        Can be called periodically or when thread is finishing.
        """
        if hasattr(self.thread_local, 'file_cache'):
            file_cache = self.thread_local.file_cache
            for handles in file_cache.values():
                offset_fptr, json_fptr = handles
                try:
                    offset_fptr.close()
                    json_fptr.close()
                except:
                    pass  # Ignore errors during cleanup
            file_cache.clear()

    def get_thread_cache_stats(self):
        """
        Get statistics about current thread's file handle cache.

        Returns:
            dict: Cache statistics including size and cached files
        """
        if not hasattr(self.thread_local, 'file_cache'):
            return {
                "cache_size": 0,
                "max_cache_size": self.max_files_per_thread,
                "cached_files": []
            }

        file_cache = self.thread_local.file_cache
        return {
            "cache_size": len(file_cache),
            "max_cache_size": self.max_files_per_thread,
            "cached_files": list(file_cache.keys())
        }

    def enumerate_doc_ids(self):
        """
        Generator that yields all available document IDs in the en00 stream.

        Note: This method is not optimized for concurrent access as it's
        typically used for dataset exploration, not high-QPS serving.

        Yields:
            str: ClueWeb22 document IDs
        """
        # Base directory for English documents (stream en00)
        base_dir = os.path.join(self.cw22_dir_txt, 'en', 'en00')

        # Pattern for offset files (e.g., en0000-31.offset)
        file_pattern = re.compile(r'(en\d{4})-(\d{2})\.offset')

        # Iterate through subdirectories (en0000, en0001, etc.)
        for subdir in sorted(os.listdir(base_dir)):
            subdir_path = os.path.join(base_dir, subdir)

            # Validate directory format
            if not os.path.isdir(subdir_path) or not re.match(r'en\d{4}', subdir):
                continue

            # Process files in each subdirectory
            for filename in sorted(os.listdir(subdir_path)):
                # Skip checksum files
                if filename.endswith('.checksum'):
                    continue

                match = file_pattern.match(filename)
                if not match:
                    continue

                # Extract file components
                file_subdir, file_seq = match.groups()

                # Verify corresponding JSON file exists
                json_file = os.path.join(subdir_path, f"{file_subdir}-{file_seq}.json.gz")
                if not os.path.exists(json_file):
                    continue

                # Calculate number of documents in this file
                offset_file = os.path.join(subdir_path, filename)
                try:
                    with open(offset_file, 'rb') as f:
                        f.seek(0, 2)  # Seek to end
                        file_size = f.tell()

                        # Validate file format (each offset is 11 bytes)
                        if file_size % 11 != 0:
                            print(f"Warning: Offset file {offset_file} size {file_size} "
                                  f"is not a multiple of 11")
                            continue

                        # Calculate document count
                        num_offsets = file_size // 11
                        num_docs = num_offsets - 1  # N offsets = N-1 documents

                        if num_docs <= 0:
                            continue

                    # Generate document IDs
                    for doc_seq in range(num_docs):
                        doc_id = f"clueweb22-{file_subdir}-{file_seq}-{doc_seq:05d}"
                        yield doc_id

                except (IOError, OSError) as e:
                    print(f"Error reading offset file {offset_file}: {e}")
                    continue


if __name__ == "__main__":
    # Example usage and testing
    doc_ids = [
        "clueweb22-en0035-22-03042",
        "clueweb22-en0036-00-17596",
        "clueweb22-en0041-84-02366",
        "clueweb22-en0038-21-02172",
        "clueweb22-en0040-20-05288"
    ]

    # Initialize with custom cache size
    cw22_docs = ClueWeb22Docs(max_files_per_thread=5)

    for docid in doc_ids:
        try:
            doc_text = cw22_docs.get_txt(docid)
            print("-------------------------------------------")
            print(f"Document ID: {docid}")
            print(f"Content type: {type(doc_text)}")
            print(f"Content preview: {doc_text[:200]}...")  # First 200 bytes

            # Show cache stats after each access
            stats = cw22_docs.get_thread_cache_stats()
            print(f"Cache: {stats['cache_size']}/{stats['max_cache_size']} files")

        except Exception as e:
            print(f"Error retrieving {docid}: {e}")

    print("\nFinal cache stats:", cw22_docs.get_thread_cache_stats())

    # Clean up cache when done (optional but recommended for long-running processes)
    cw22_docs.cleanup_thread_cache()
