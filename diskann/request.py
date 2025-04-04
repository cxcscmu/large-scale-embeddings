import requests
import numpy as np
import time
import threading

lock = threading.Lock()

# URL = "http://10.1.1.26:8008"
K = 10
Ls = 256
PORT = 8008
QUERY_FILE = (
    "/home/karrym/data/ann_index/embeds/clueweb/MiniCPM-Embedding-Light/queries/marcoweb_minicpm-light_queries_test.bin"
)

SHARDS = [
    "0-3",
    "4-7",
    "8-11",
    "12-15",
]

PREFIX = {
    "0-3": "marcoweb_minicpm-light_index_0-3_R80_L120_B64_M64_T8",  # boston-1-9
    "4-7": "marcoweb_minicpm-light_index_4-7_R80_L120_B64_M80_T16",  # boston-1-24
    "8-11": "marcoweb_minicpm-light_index_8-11_R80_L120_B64_M80_T16",  # boston-1-10
    "12-15": "marcoweb_minicpm-light_index_12-15_R80_L120_B64_M80_T16",  # boston-1-23
}

HOST = {
    "0-3": "10.1.1.26",  # boston-1-9
    "4-7": "10.1.1.18",  # boston-1-24
    "8-11": "10.1.1.25",  # boston-1-10
    "12-15": "10.1.1.19",  # boston-1-23
}

OFFSET = {
    "0-3": 0,
    "4-7": 25231240,
    "8-11": 50462480,
    "12-15": 75693720,
}


def read_fbin(filename):
    data = np.fromfile(filename, dtype=np.float32)
    # print(f"fbin.shape: {data.shape}")
    shape = data.view(np.int32)[:2]
    print(f"shape: {shape}")
    data = data[2:].reshape(shape)
    return data


def rerank(I, D):
    order = np.argsort(D, axis=1)[:, ::-1]
    I = np.take_along_axis(I, order, axis=1)
    D = np.take_along_axis(D, order, axis=1)
    return I, D


def request_shard(shard, indices, distances, i):
    # print(f"shard: {shard}")
    url = f"http://{HOST[shard]}:{PORT}"
    # print(f"URL: {url}")
    response = requests.post(url, json=jsonquery).json()
    # print(response)
    with lock:
        indices[i] = response["indices"]
        distances[i] = response["distances"]
    # print(f"shard {shard} finished")


OFFSET_ARRAY = []
for shard in SHARDS:
    OFFSET_ARRAY.extend([OFFSET[shard]] * K)
OFFSET_ARRAY = np.array(OFFSET_ARRAY)
# print(f"OFFSET_ARRAY: {OFFSET_ARRAY.shape}")


N_TEST = 100
queries = read_fbin(QUERY_FILE)[:N_TEST]
start_time = time.perf_counter()
for query_id in range(queries.shape[0]):
    query = queries[query_id]
    jsonquery = {
        "Ls": Ls,
        "query_id": query_id,
        "query": queries[0].tolist(),
        "k": K,
    }
    indices = [None] * len(SHARDS)
    distances = [None] * len(SHARDS)
    threads = []
    for i, shard in enumerate(SHARDS):
        thread = threading.Thread(target=request_shard, args=(shard, indices, distances, i))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    I = np.array(indices).reshape(1, -1) + OFFSET_ARRAY
    D = np.array(distances).reshape(1, -1)
    I, D = rerank(I, D)


end_time = time.perf_counter()
average_time = (end_time - start_time) / N_TEST
qps = 1 / average_time
print(f"Total time: {end_time - start_time} seconds")
print(f"QPS: {qps}")
print(f"Average time per query: {average_time} seconds")
