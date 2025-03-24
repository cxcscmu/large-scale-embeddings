import numpy as np
import os 
import pickle


def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        print("number of queries: ", nvecs)
        print("dimension: ", dim)
        f.seek(4+4)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def retrieval_result_read(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    print("n: ", n)
    print("d: ", d)
    assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    f = open(fname, "rb")
    f.seek(4+4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D


def write_embed_to_binary(embeddings, output_path): 
    num, dim = embeddings.shape
    with open(output_path, "wb") as f:
        f.write(num.to_bytes(4, 'little'))
        f.write(dim.to_bytes(4, 'little'))
        f.write(embeddings.tobytes())


def write_docids_to_pkl(docids, output_path): 
    with open(output_path, "wb") as f:
        pickle.dump(docids, f)

    
def convert_encoded_pkl_to_binary(input_path, embed_output_path, docid_output_path): 
    with open(input_path, "rb") as f:
        embeds, docids = pickle.load(f)
    write_embed_to_binary(embeds, embed_output_path)
    write_docids_to_pkl(docids, docid_output_path)


def convert_encoded_pkls_to_binary(input_dir, input_names, embed_output_path, docid_output_path): 
    
    embeds = []
    ids = []
    for file in input_names: 
        with open(os.path.join(input_dir, file), "rb") as f:
            shard_embeds, shard_ids = pickle.load(f)
        # add to the embed part
        embeds.append(shard_embeds)
        ids.extend(shard_ids)

    embeds = np.concatenate(embeds)
    write_embed_to_binary(embeds, embed_output_path)
    write_docids_to_pkl(ids, docid_output_path)


