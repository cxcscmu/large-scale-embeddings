import numpy as np
import os 
import pickle




def read_trec_qrels(path): 
    qid_qrels = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            qid = parts[0]
            qrel = parts[2]
            qid_qrels[qid] = qrel
    return qid_qrels


def get_qrels_array(qid_qrels_dict): 
    qrels = []
    for qid in qid_qrels_dict: 
        qrels.append(qid_qrels_dict[qid])
    return np.array(qrels)[:, None]


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


def retrieval_result_read(fname, e2e=False):
    """
    Read the binary ground truth file in DiskANN format. 
    If e2e is given as True, no distances array will be read (end of end qrel scenario). 
    """
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    print("n: ", n)
    print("d: ", d)
    # validity check 
    if e2e: 
        assert os.stat(fname).st_size == 8 + n * d * 4
    else: 
        assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    
    f = open(fname, "rb")
    f.seek(4+4)

    I, D = None, None
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    if not e2e: 
        D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)

    return I, D


def write_embed_to_binary(embeddings, output_path): 
    """
    Write the embedding array into a binary file in ANN-Indexing (DiskANN, SPTAG) format. 
    The content of the output file can be access through: embeds = read_fbin(output_path)
    """
    num, dim = embeddings.shape
    with open(output_path, "wb") as f:
        f.write(num.to_bytes(4, 'little'))
        f.write(dim.to_bytes(4, 'little'))
        f.write(embeddings.tobytes())


def write_qrels_to_binary(qid_qrels_dict, output_binary_path, docid_one_index=False): 
    """
    Write the qrels given in a dictionary to ANN-Indexing (DiskANN, SPTAG) binary format. 
    Currently only 1 qrel per query allowed. 
    The content of the output file can be access through: I, _ = retrieval_result_read(output_binary_path, e2e=True)
    """
    nrows = len(qid_qrels_dict)
    ncols = 1

    qrels = get_qrels_array(qid_qrels_dict)
    # convert to 0-indexing if needed as DiskANN arranged by candidate index in corpus 
    if docid_one_index: 
        qrels -= 1

    with open(output_binary_path, "wb") as f:
        offset = 0
        # header
        f.write(nrows.to_bytes(4, 'little')) # number of points
        f.write(ncols.to_bytes(4, 'little')) # dimension
        offset += 8
        
        # ids
        f.seek(offset)
        f.write(qrels.astype('uint32').tobytes()) 
        offset += qrels.nbytes


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


