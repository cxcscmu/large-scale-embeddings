import gzip
import os
import re
import json


class ClueWeb22Docs:
    """
    Fetch ClueWeb22 documents. Currently only supports the txt file
    hierarchy. Upgrading too support the html (warc.gz) hierarchy
    would be easy.

    Usage:
        cw22_docs('/bos/tmp1/ClueWeb22_L')

    	.get_txt('clueweb22-en0102-03-00004')
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
        Prepare to access ClueWeb22 documents.

        cw22_dir_root:	For example, '/bos/tmp1/ClueWeb22_L'
        """

        # Location of ClueWeb22 files.
        self.cw22_dir_txt = os.path.join(cw22_dir_root, 'txt')

        # Files do not close after a read in case the next lookup is
        # from the same files. These variables describe the open files.
        self._fn_subdir = None  # E.g., en0102
        self._fn_file_seq = None  # E.g., 03
        self._fptr_json = None  # File ptr
        self._fptr_offset = None  # File ptr

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

    def _open_files(self, lang, stream, subdir, file_seq):
        """
        Return file pointers for a .json.gz file and its .offset file.
        The files may be open from a previous access.
        """

        # If an open json file does not contain the document...
        if ((self._fn_subdir != subdir) or
                (self._fn_file_seq != file_seq)):

            # Close any open files
            if self._fptr_json != None:
                self._fptr_json.close()
                self._fptr_offset.close()

            # Open the right files
            path = os.path.join(self.cw22_dir_txt, lang, stream, subdir,
                                f'{subdir}-{file_seq}')
            self._fptr_json = open(f'{path}.json.gz', 'rb')
            self._fptr_offset = open(f'{path}.offset', 'rb')
            self._fn_subdir = subdir
            self._fn_file_seq = file_seq

        return ((self._fptr_offset, self._fptr_json))

    def get_txt(self, docid):
        """Get the txt representation of a ClueWeb22 document"""

        # Parse the ClueWeb22 docid
        matches = ClueWeb22Docs.cwid_regex.search(docid)
        lang, stream, subdir, file_seq, doc_seq = matches.group(3, 2, 1, 6, 7)

        # Open the .offset and .json.gz files if they are not open already
        fptr_offset, fptr_json = \
            self._open_files(lang, stream, subdir, file_seq)

        # Extract the document
        doc_start, doc_end = self._get_offsets(fptr_offset, doc_seq)
        fptr_json.seek(doc_start, 0)
        doc = fptr_json.read(doc_end - doc_start)
        doc = gzip.decompress(doc)

        return doc

    def enumerate_doc_ids(self):
        # Base directory for en00 subdirectories
        base_dir = os.path.join(self.cw22_dir_txt, 'en', 'en00')

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

# if __name__ == "__main__":
#     # Print the first 10 document IDs as an example
#     cw2_docs = ClueWeb22Docs()
#     last_id = None
#     prev_partition_id = None
#     cnt = 0
#
#     for i, doc_id in enumerate(cw2_docs.enumerate_doc_ids()):
#         partition_id = doc_id.split("-")[2]
#         if partition_id != prev_partition_id:
#             print(cnt, last_id)
#             cnt = 0
#         else:
#             cnt += 1
#         prev_partition_id = partition_id
#         last_id = doc_id
#
#         if partition_id == "99":
#             break

# if __name__ == "__main__":
#     doc_ids = ["clueweb22-en0035-22-03042",
#                "clueweb22-en0036-00-17596",
#                "clueweb22-en0041-84-02366",
#                "clueweb22-en0038-21-02172",
#                "clueweb22-en0040-20-05288"]
#     cw2_docs = ClueWeb22Docs()
#
#     for docid in doc_ids:
#         doc_text = cw2_docs.get_txt(docid)
#         print("-------------------------------------------")
#         print(docid)
#         print(doc_text)
