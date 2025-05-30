import pickle
import numpy as np
import os
import sys
from tqdm import tqdm


def write_embed_to_binary(embeddings, output_path):
    """
    Write the embedding array into a binary file in ANN-Indexing (DiskANN, SPTAG) format.
    The content of the output file can be accessed through: embeds = read_fbin(output_path)

    Args:
        embeddings: numpy array of shape (num, dim)
        output_path: path to save the binary file
    """
    num, dim = embeddings.shape

    with open(output_path, "wb") as f:
        # Write header information
        f.write(num.to_bytes(4, "little"))
        f.write(dim.to_bytes(4, "little"))

        # Use tqdm progress bar to track writing progress
        chunk_size = 10000  # Number of embedding vectors to process at once
        with tqdm(total=num, desc="Writing embeddings to binary") as pbar:
            for i in range(0, num, chunk_size):
                end_idx = min(i + chunk_size, num)
                # Write the current batch of embedding vectors
                f.write(embeddings[i:end_idx].tobytes())
                pbar.update(end_idx - i)


def write_docids_to_pkl(docids, output_path):
    """
    Write document IDs to a pickle file.

    Args:
        docids: list or numpy array of document IDs
        output_path: path to save the pickle file
    """
    # Convert to Python list if it's a numpy array
    if isinstance(docids, np.ndarray):
        docids = docids.tolist()

    with open(output_path, "wb") as f:
        pickle.dump(docids, f)


def load_pickle_in_chunks(file_path, chunk_size=1000000):
    """
    Load pickle file in chunks to reduce memory usage.

    Args:
        file_path: path to the pickle file
        chunk_size: number of embedding vectors to load at once

    Returns:
        generator yielding (chunk_embeds, chunk_docids)
    """
    # Get file size for information
    file_size = os.path.getsize(file_path)
    print(f"Total file size: {file_size / (1024**3):.2f} GB")

    with open(file_path, "rb") as f:
        try:
            # Load the full object once to get metadata
            data = pickle.load(f)
            # Check if the data is a list with 2 elements (not a tuple as previously assumed)
            if isinstance(data, list) and len(data) == 2:
                embeds, docids = data
                del data  # Free memory

                # Get embedding shape information
                num, dim = embeds.shape
                print(f"Embeddings shape: ({num}, {dim})")

                # Process in chunks
                total_chunks = (num + chunk_size - 1) // chunk_size
                for i in tqdm(range(total_chunks), desc="Processing chunks"):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, num)
                    # Yield current batch of embeddings and document IDs
                    yield embeds[start_idx:end_idx], docids[start_idx:end_idx]

                # Clean up memory
                del embeds
                del docids
            else:
                raise ValueError("Pickle file does not contain expected list of [embeds, docids]")
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            raise


def convert_encoded_pkl_to_binary_in_chunks(input_path, embed_output_path, docid_output_path, chunk_size=1000000):
    """
    Convert pickle format embeddings to binary format in chunks.

    Args:
        input_path: path to the input pickle file
        embed_output_path: path to save the embeddings binary file
        docid_output_path: path to save the document IDs pickle file
        chunk_size: number of embedding vectors to process at once
    """
    try:
        # First get basic file information
        with open(input_path, "rb") as f:
            # Only read the header information to get shape
            data = pickle.load(f)
            if not isinstance(data, list) or len(data) != 2:
                raise ValueError("Invalid pickle file format - expected a list with 2 elements")

            embeds, docids = data
            num, dim = embeds.shape

            # Keep a reference to all docids or create a copy if needed
            all_docids = docids

            # Free memory
            del data
            del embeds
            del docids
    except Exception as e:
        print(f"Error during initial file inspection: {e}")
        raise

    # Prepare binary file header
    with open(embed_output_path, "wb") as f:
        f.write(num.to_bytes(4, "little"))
        f.write(dim.to_bytes(4, "little"))

    # Process embedding vectors in chunks
    chunk_count = 0
    with tqdm(total=num, desc="Converting embeddings") as pbar:
        for chunk_embeds, chunk_docids in load_pickle_in_chunks(input_path, chunk_size):
            # Append embedding vectors to binary file
            with open(embed_output_path, "ab") as f:
                f.write(chunk_embeds.tobytes())

            chunk_count += len(chunk_embeds)
            pbar.update(len(chunk_embeds))

    # Write all document IDs
    print(f"Writing {len(all_docids)} document IDs to {docid_output_path}")
    write_docids_to_pkl(all_docids, docid_output_path)
    print(f"Conversion complete. Processed {chunk_count} embeddings.")


if __name__ == "__main__":
    # get the first argument from the command line
    shard_id = sys.argv[1]
    print(f"Processing shard ID: {shard_id}")

    # TODO: Update these paths as needed
    input_path = f"/bos/tmp6/jmcoelho/embeddings/minicpm-embedding-light/fineweb/sharded/fineweb-CC-MAIN-2024-51.{shard_id}.merged.pkl"
    embed_output_path = f"/bos/tmp2/jening/fineweb_embeddings/bin/fineweb.{shard_id}.bin"
    docid_output_path = f"/bos/tmp2/jening/fineweb_embeddings/docids/fineweb.docids.{shard_id}.pkl"

    # Use smaller chunk size to reduce memory usage
    # Adjust this value based on available memory
    chunk_size = 100000

    convert_encoded_pkl_to_binary_in_chunks(input_path, embed_output_path, docid_output_path, chunk_size=chunk_size)
