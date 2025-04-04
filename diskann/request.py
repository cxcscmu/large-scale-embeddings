import requests
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def request_shard(shard, jsonquery):
    # print(f"shard: {shard}")
    url = f"http://{HOST[shard]}:{PORT}"
    # print(f"URL: {url}")
    try:
        response = requests.post(url, json=jsonquery)
        response_dict = response.json()
    except Exception as e:
        print(response)
        raise e
    return response_dict["indices"], response_dict["distances"]
    # print(f"shard {shard} finished")


OFFSET_ARRAY = []
for shard in SHARDS:
    OFFSET_ARRAY.extend([OFFSET[shard]] * K)
OFFSET_ARRAY = np.array(OFFSET_ARRAY)
# print(f"OFFSET_ARRAY: {OFFSET_ARRAY.shape}")

N_TEST = 500
queries = read_fbin(QUERY_FILE)[:N_TEST]
start_time = time.perf_counter()
indices = {}
distances = {}
futures = []
time0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=50) as executor:
    for query_id in range(queries.shape[0]):
        query = queries[query_id]
        jsonquery = {
            "Ls": Ls,
            "query_id": query_id,
            "query": queries[0].tolist(),
            "k": K,
        }
        indices[query_id] = []
        distances[query_id] = []
        futures.extend([executor.submit(request_shard, shard, jsonquery) for shard in SHARDS])

    for future in as_completed(futures):
        result = future.result()
        query_id = result[0]
        indices[query_id].extend(result[1])
        distances[query_id].extend(result[2])


time1 = time.perf_counter()

for query_id in range(queries.shape[0]):
    assert len(indices[query_id]) == K * len(SHARDS)
    I = np.array(indices[query_id]).reshape(1, -1) + OFFSET_ARRAY
    D = np.array(distances[query_id]).reshape(1, -1)
    I, D = rerank(I, D)

time2 = time.perf_counter()

io_time = time1 - time0
compute_time = time2 - time1
end_time = time.perf_counter()
total_time = end_time - start_time
average_time = (end_time - start_time) / N_TEST
qps = 1 / average_time
print(f"Total time: {total_time} seconds")
print(f"QPS: {qps}")
print(f"Average time per query: {average_time} seconds")
print(f"IO time: {io_time} seconds ({io_time / total_time * 100:.2f}%)")
print(f"Compute time: {compute_time} seconds ({compute_time / total_time * 100:.2f}%)")
