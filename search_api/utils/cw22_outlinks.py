import gzip
import os
import re


class ClueWeb22Outlinks:
    """
    Fetch ClueWeb22 documents. Currently only supports the outlink file
    hierarchy. Upgrading too support the html (warc.gz) hierarchy
    would be easy.

    Usage:
        cw22_docs('/bos/tmp1/ClueWeb22_L')

    	.get_outlink('clueweb22-en0102-03-00004')
    """

    # ------------------ Global variables ---------------------- #

    cwid_regex = re.compile(
        "^clueweb22-"
        "((((?:de)|(?:en)|(?:es)|(?:fr)|(?:it)|(?:ja)|(?:nl)|(?:pl)|(?:pt)"
        "|(?:zh_chs)|(?:other))"
        "([0-9]{2}))"
        "([0-9]{2}))-"
        "([0-9]{2})-"
        "([0-9]{5})$"
    )

    # ------------------ Methods (alphabetical order) ---------- #

    def __init__(self, cw22_dir_root='/bos/tmp1/ClueWeb22_B'):
        """
        Initialize ClueWeb22 outlinks fetcher.

        Args:
            cw22_dir_root: Root directory of ClueWeb22 files (e.g., '/bos/tmp1/ClueWeb22_B')
        """
        # Store path to outlink directory (read-only, thread-safe)
        self.cw22_dir_outlink = os.path.join(cw22_dir_root, 'outlink')

    def __str__(self):
        """Human readable information about class instances."""
        return (str(self.__dict__))

    def _get_offsets(self, fptr_offset, doc_seq):
        """
        Get the starting and ending byte offsets of a document from
        an open .offsets file.
        """

        cw22_offset_length = 11  # 10 characters per offset + newline

        fptr_offset.seek(int(doc_seq) * cw22_offset_length, 0)
        doc_start = int(fptr_offset.read(cw22_offset_length - 1))
        fptr_offset.read(1)  # skip newline
        doc_end = int(fptr_offset.read(cw22_offset_length - 1))

        return ((doc_start, doc_end))

    def get_outlink(self, docid):
        """
        Get the outlink representation of a ClueWeb22 document.

        Thread-safe implementation: Each call opens files independently,
        reads the required data, and immediately closes files.

        Args:
            docid: ClueWeb22 document ID (e.g., 'clueweb22-en0102-03-00004')

        Returns:
            bytes: Decompressed outlink content

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

        # Build file paths
        base_path = os.path.join(self.cw22_dir_outlink, lang, stream, subdir,
                                 f'{subdir}-{file_seq}')
        json_path = f'{base_path}.json.gz'
        offset_path = f'{base_path}.offset'

        try:
            # Step 1: Read offset information (independent file handle)
            with open(offset_path, 'rb') as offset_fptr:
                doc_start, doc_end = self._get_offsets(offset_fptr, doc_seq)

            # Step 2: Read compressed document data (independent file handle)
            with open(json_path, 'rb') as json_fptr:
                json_fptr.seek(doc_start, 0)
                compressed_doc = json_fptr.read(doc_end - doc_start)

            # Step 3: Decompress and return (no file I/O, thread-safe)
            return gzip.decompress(compressed_doc)

        except (IOError, OSError) as e:
            raise IOError(f"Failed to read outlink document {docid}: {e}")

    def enumerate_doc_ids(self):
        # Base directory for en00 subdirectories
        base_dir = os.path.join(self.cw22_dir_outlink, 'en', 'en00')

        # Pattern for offset files (like en0000-31.offset)
        file_pattern = re.compile(r'(en\d{4})-(\d{2})\.offset')

        # Iterate through subdirectories (en0000, en0001, etc.)
        for subdir in sorted(os.listdir(base_dir)):
            subdir_path = os.path.join(base_dir, subdir)

            # Check if it's a directory and has the right format (en0000, en0001, etc.)
            if not os.path.isdir(subdir_path) or not re.match(r'en\d{4}', subdir):
                continue

            # Iterate through files in the subdirectory
            for filename in sorted(os.listdir(subdir_path)):
                # Skip checksum files
                if filename.endswith('.checksum'):
                    continue

                match = file_pattern.match(filename)
                if not match:
                    continue

                # Get subdir and file_seq from the filename
                file_subdir, file_seq = match.groups()

                # Check if the corresponding JSON.gz file exists
                json_file = os.path.join(subdir_path, f"{file_subdir}-{file_seq}.json.gz")
                if not os.path.exists(json_file):
                    continue

                # Determine the number of documents in this file
                offset_file = os.path.join(subdir_path, filename)
                with open(offset_file, 'rb') as f:
                    # Get file size
                    f.seek(0, 2)
                    file_size = f.tell()

                    # Each offset is 11 bytes (10 digits + newline)
                    if file_size % 11 != 0:
                        print(f"Warning: Offset file {offset_file} size {file_size} is not a multiple of 11")
                        continue

                    # Calculate the number of offsets
                    num_offsets = file_size // 11

                    # The number of documents is one less than the number of offsets
                    # (since each document spans from one offset to the next)
                    num_docs = num_offsets - 1

                    if num_docs <= 0:
                        continue

                # Generate document IDs
                for doc_seq in range(num_docs):
                    # Format: clueweb22-en0000-00-00000
                    doc_id = f"clueweb22-{file_subdir}-{file_seq}-{doc_seq:05d}"
                    yield doc_id
