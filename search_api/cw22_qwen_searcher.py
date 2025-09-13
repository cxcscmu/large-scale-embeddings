import pickle
import random
import string
import time
from collections import Counter
from contextlib import contextmanager

import numpy as np
import requests
from tqdm import tqdm

from utils.cw22_files import ClueWeb22Docs
from utils.cw22_outlinks import ClueWeb22Outlinks
from utils.performance_monitor import PerformanceMonitor
from utils.query_encoder import QueryEncoder


class ClueWeb22Searcher:
    """
    A distributed search engine for ClueWeb22 dataset that performs semantic search 
    across multiple shards using vector embeddings.

    This class handles:
    1. Query encoding using a neural model
    2. Distributed search across multiple server shards
    3. Result aggregation and reranking
    4. Document ID translation and text retrieval
    """

    def __init__(self, enable_monitoring=True):
        """
        Initialize the ClueWeb22Searcher with necessary components.

        Args:
            verbose (bool): If True, prints detailed information during search operations
            enable_monitoring (bool): If True, enables performance monitoring
        """
        print("---ClueWeb22Searcher Initializing---")

        # Initialize performance monitoring
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.monitor = PerformanceMonitor(service_name="ClueWeb22", report_interval=60, window_size=1000)
            self.monitor.start()
            print("Performance monitoring enabled (60s reports)")

        # Define endpoints for distributed search servers (shards)
        self.distributed_indices = {0: "http://10.1.1.23:51200/search?",
                                    1: "http://10.1.1.22:51200/search?",
                                    2: "http://10.1.1.19:51200/search?",
                                    3: "http://10.1.1.24:51200/search?"}

        # Load document retrieval module
        self.cw2_docs = ClueWeb22Docs()
        self.cw22_outlinks = ClueWeb22Outlinks()
        print("---ClueWeb22Searcher Init Step 1: ClueWeb22 Texts Loaded---")

        # Initialize query encoder for transforming text queries into vector embeddings
        self.query_encoder = QueryEncoder(model_name="Qwen/Qwen3-Embedding-0.6B")
        self.query_encoder.encode_query("Hello ClueWeb22!")  # Warm-up the encoder
        print("---ClueWeb22Searcher Init Step 2: Query Encoder Loaded---")

        # Load document ID mappings for each shard
        # These map internal IDs to actual ClueWeb22 document IDs
        self.docid_map = dict()
        for i in range(4):
            input_path = f"/bos/tmp2/jening/clueweb22-en-a-qwen3-embedding-0.6b/en00/docids/cw22-a.docids.{i}.pkl"
            with open(input_path, "rb") as f:
                self.docid_map[i] = pickle.load(f)

            print(f"docid_map {i} loaded")

        print("---ClueWeb22Searcher Init Step 3: Docid Mappings Loaded---")

        # Directory for storing query embeddings
        self.query_emb_dir = "/bos/usr0/jening/search_service/query_embeddings/"
        self.query_counter = 0

        self.session_id = random.randint(1, 2147483647)
        self.session = requests.Session()
        print(f"---ClueWeb22Searcher Init Step 4: Search session id {self.session_id}---")

        # Track distribution of results across shards
        self.shard_distribution = Counter()

    def _monitor_context(self, component_name):
        """Helper method to get monitoring context or dummy context."""
        if self.enable_monitoring:
            return self.monitor.measure(component_name)
        else:
            return self._dummy_context()

    @contextmanager
    def _dummy_context(self):
        """Dummy context manager when monitoring is disabled."""
        yield

    def _encode_query_text(self, query_text):
        """
        Encode a text query into a vector embedding.

        Args:
            query_text (str): The query text to encode

        Returns:
            numpy.ndarray: Query embedding vector
        """
        with self._monitor_context("query_encoding"):
            q_emb = self.query_encoder.encode_query(query_text)
            qid = self.query_counter
            self.query_counter += 1
            return q_emb

    def _from_precomputed_emb(self, q_emb):
        """
        Handle precomputed embedding (legacy method).

        Args:
            q_emb: Precomputed query embedding

        Returns:
            int: Query ID assigned to this query
        """
        qid = self.query_counter
        self.query_counter += 1

        # Save query embedding to be used by the distributed search servers
        np.save(file=self.query_emb_dir + f"q_emb_{self.session_id}_{qid}.npy",
                arr=q_emb)
        return qid

    def _translate_ids(self, raw_docs):
        """
        Translate internal document IDs to actual ClueWeb22 document IDs while
        preserving distance scores.

        Args:
            raw_docs (list): List of tuples (distance, raw_id, shard_id)

        Returns:
            list: List of tuples (distance, docid) with translated ClueWeb22 document IDs
                  and their corresponding similarity scores
        """
        with self._monitor_context("id_translation"):
            return [(dist, self.docid_map[shard_id][raw_id]) for dist, raw_id, shard_id in raw_docs]

    def search(self,
               query_text,
               k=10,
               complexity=None,
               num_of_shards=4,
               with_distance=False,
               from_precomputed_emb=False,
               parallel=True,
               cw22_a=False):
        """
        Perform distributed search across multiple shards for the given query.

        Args:
            query_text (str): The text query to search for
            k (int): Number of top results to return
            complexity (int): Complexity parameter for the search
            num_of_shards (int): Number of shards to search across
            with_distance (bool): If True, return document IDs with their similarity scores
            from_precomputed_emb (bool): If True, use a precomputed embedding
            parallel (bool): If True, query shards in parallel using ThreadPoolExecutor
            cw22_a (bool): If False, search only shards 0-3; if True, search all 40 shards

        Returns:
            list: If with_distance=False, returns top-k document IDs matching the query
                 If with_distance=True, returns top-k tuples (similarity_score, document_id)
        """
        raw_docs = list()

        # Step 1: Query encoding
        if from_precomputed_emb:
            q_emb = query_text.tolist()
        else:
            q_emb = self._encode_query_text(query_text).tolist()

        # Set default complexity
        if complexity is None:
            complexity = 5 * k

        shard_ids = range(4)

        def query_shard(shard_id, q_emb, complexity):
            """Helper function to query a single shard"""
            with self._monitor_context(f"shard_{shard_id}"):
                # Skip shard if not valid
                if shard_id not in self.distributed_indices:
                    return []

                # Prepare payload for POST request
                payload = {
                    "q_emb": q_emb,
                    "k": k,
                    "complexity": complexity
                }

                # Send POST request with embedding in body
                url = self.distributed_indices[shard_id]
                response = self.session.post(url, json=payload)

                shard_results = []
                if response.status_code == 200:
                    data = response.json()
                    this_ids = data["indices"]
                    this_dist = data["distances"]
                    this_len = len(this_ids)

                    # Store results as (distance, raw_id, shard_id) tuples
                    for i in range(this_len):
                        shard_results.append((this_dist[i], this_ids[i], shard_id))

                else:
                    print("Error at node", shard_id, response.text)

                return shard_results

        # Step 2: Distributed search
        with self._monitor_context("distributed_search"):
            if parallel:
                # Import ThreadPoolExecutor here to minimize changes
                from concurrent.futures import ThreadPoolExecutor

                # Use ThreadPoolExecutor to query shards in parallel
                max_concurrent = min(len(shard_ids), 16)
                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    # Submit tasks for each shard and collect results
                    future_to_shard = {executor.submit(query_shard, shard_id, q_emb, complexity): shard_id
                                       for shard_id in shard_ids}

                    # Collect results as they complete
                    for future in future_to_shard:
                        raw_docs.extend(future.result())
            else:
                # Original sequential implementation
                for shard_id in range(num_of_shards):
                    raw_docs.extend(query_shard(shard_id, q_emb, complexity))

        # Step 3: Result aggregation and reranking
        with self._monitor_context("result_aggregation"):
            raw_docs.sort(key=lambda x: x[0], reverse=True)
            raw_docs = raw_docs[:k]  # Keep only top-k results

            # Optional: Update analytics on which shards are returning results
            self.shard_distribution.update([shard_id for _, _, shard_id in raw_docs])

        # Step 4: Translate internal IDs to actual ClueWeb22 document IDs
        docs = self._translate_ids(raw_docs)

        # Return either (distance, docid) tuples or just docids based on with_distance parameter
        if with_distance:
            return docs

        return [docid for _, docid in docs]

    def get_doc_texts(self, docids):
        """
        Retrieve the actual document text content for the given document IDs.

        Args:
            docids (list): List of ClueWeb22 document IDs

        Returns:
            list: Document texts corresponding to the given IDs
        """
        with self._monitor_context("document_retrieval"):
            doc_texts = list()
            for docid in docids:
                doc_text = self.cw2_docs.get_txt(docid)
                doc_texts.append(doc_text)
            return doc_texts

    def get_doc_texts_parallel(self, docids, thread_pool):
        """
        Get document texts using thread pool for parallel I/O
        """
        with self._monitor_context("document_retrieval"):
            if not docids:
                return []

            def safe_get_txt(docid):
                """Thread-safe document retrieval from CW22"""
                try:
                    doc_text = self.cw2_docs.get_txt(docid)
                    return doc_text if doc_text is not None else b""  # Handle None
                except Exception as e:
                    return b""

            from concurrent.futures import as_completed

            # Submit all docid fetch tasks to the thread pool
            future_to_docid = {
                thread_pool.submit(safe_get_txt, docid): docid
                for docid in docids
            }

            # Collect results preserving original docid order
            docid_to_text = {}
            for future in as_completed(future_to_docid):
                docid = future_to_docid[future]
                docid_to_text[docid] = future.result()

            return [docid_to_text.get(docid, b"") for docid in docids]

    def get_outlinks(self, docids):
        """
        Retrieve the actual document outlinks content for the given document IDs.

        Args:
            docids (list): List of ClueWeb22 document IDs

        Returns:
            list: Document outlinks corresponding to the given IDs
        """
        with self._monitor_context("outlinks_retrieval"):
            outlinks = list()
            for docid in docids:
                doc_outlink = self.cw22_outlinks.get_outlink(docid)
                outlinks.append(doc_outlink)
            return outlinks

    def get_outlinks_parallel(self, docids, thread_pool):
        """
        Get document outlinks using thread pool for parallel I/O
        """
        with self._monitor_context("outlinks_retrieval"):
            if not docids:
                return []

            def safe_get_outlink(docid):
                """Thread-safe outlink retrieval from CW22"""
                try:
                    return self.cw22_outlinks.get_outlink(docid)
                except Exception as e:
                    return ""

            from concurrent.futures import as_completed

            # Submit all outlink fetch tasks to the thread pool
            future_to_docid = {
                thread_pool.submit(safe_get_outlink, docid): docid
                for docid in docids
            }

            # Collect results preserving original docid order
            docid_to_outlink = {}
            for future in as_completed(future_to_docid):
                docid = future_to_docid[future]
                docid_to_outlink[docid] = future.result()

            return [docid_to_outlink.get(docid, "") for docid in docids]

    def close(self):
        """Clean up resources and stop monitoring."""
        if hasattr(self, 'session'):
            self.session.close()

        if self.enable_monitoring:
            self.monitor.stop()
            print("Performance monitoring stopped")


def run_trec_evaluation(k, complexity):
    """
    Run an evaluation using TREC queries and calculate MRR@k metrics.
    This is currently commented out in the main block but shows how to use
    the searcher for an actual evaluation task.
    """
    from relevance_judgement.relevance_metric import RMetric
    searcher = ClueWeb22Searcher()

    print(f"Running trec_evaluation with k={k} & complexity={complexity}")

    import csv
    from tqdm import tqdm

    retrieved_dict = dict()

    query_counter = 0
    # Load queries from TSV file
    queries_path = "data/researchy_questions/queries_test_cleaned.tsv"
    with open(queries_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader):
            qid = row[0].strip()
            query_text = row[1].strip()
            # Search and retrieve documents for each query
            docids = searcher.search(query_text, k=k, complexity=complexity)
            # doc_texts = searcher.get_doc_texts(docids)
            retrieved_dict[qid] = docids

            # query_counter += 1
            # if query_counter >= 100:
            #     break

    # Evaluate using MRR@k metric
    r_metric = RMetric(qrels_path="data/researchy_questions/qrels_test_cleaned.tsv")
    r_metric.evaluate_all_docid(retrieved_dict, k=10, verbose=False)
    r_metric.evaluate_all_docid(retrieved_dict, k=30, verbose=False)
    r_metric.evaluate_all_docid(retrieved_dict, k=100, verbose=False)


def run_ground_truth_trec_evaluation():
    from relevance_judgement.relevance_metric import RMetric
    searcher = ClueWeb22Searcher()

    print(f"Running ground_truth_trec_evaluation")

    import csv
    from tqdm import tqdm

    retrieved_dict = dict()

    ground_truth_path = "data/researchy_questions/anns/qrels_test_anns_100.tsv"
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader):
            qid = row[0].strip()
            doc_id = row[1].strip()
            if qid not in retrieved_dict:
                retrieved_dict[qid] = list()
            retrieved_dict[qid].append(doc_id)

    for qid in tqdm(retrieved_dict):
        retrieved_dict[qid] = searcher.get_doc_texts(retrieved_dict[qid])

    # Evaluate using MRR@k metric
    r_metric = RMetric(qrels_path="data/researchy_questions/qrels_test_cleaned.tsv")
    r_metric.evaluate_all(retrieved_dict, k=10, verbose=False)
    r_metric.evaluate_all(retrieved_dict, k=30, verbose=False)
    r_metric.evaluate_all(retrieved_dict, k=100, verbose=False)


def run_anns_gt_evaluation(k, complexity):
    """
    Run an evaluation using TREC queries and calculate MRR@k metrics.
    This is currently commented out in the main block but shows how to use
    the searcher for an actual evaluation task.
    """
    from relevance_judgement.relevance_metric import RMetric
    searcher = ClueWeb22Searcher(verbose=False)

    print(f"Running anns_gt eval with k={k} & complexity={complexity}")

    from tqdm import tqdm

    def read_fbin(filename, start_idx=0, chunk_size=None):
        with open(filename, "rb") as f:
            nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
            print("number of queries: ", nvecs)
            print("dimension: ", dim)
            f.seek(4 + 4)
            nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
            arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                              offset=start_idx * 4 * dim)
        return arr.reshape(nvecs, dim)

    def read_pkl_query_ids(input_path):
        with open(input_path, "rb") as f:
            query_ids = pickle.load(f)
        return query_ids

    retrieved_dict = dict()

    query_counter = 0

    # Load queries from file
    query_id_path = "/bos/usr0/jening/PycharmProjects/DiskANN_Search/data/researchy_questions/q_emb/query_ids.pkl"
    query_ids = read_pkl_query_ids(query_id_path)

    queries_path = "/bos/usr0/jening/PycharmProjects/DiskANN_Search/data/researchy_questions/q_emb/queries_test_cleaned.bin"
    query_arr = read_fbin(queries_path)

    for idx, qid in tqdm(enumerate(query_ids)):
        qid = str(qid)

        query_emb = query_arr[idx]

        # Search and retrieve documents for each query
        docids = searcher.search(query_emb, k=k, complexity=complexity, from_precomputed_emb=True)

        doc_texts = searcher.get_doc_texts(docids)
        retrieved_dict[qid] = doc_texts

        # query_counter += 1
        # if query_counter >= 100:
        #     break

    # Evaluate using MRR@k metric
    r_metric = RMetric(qrels_path="data/researchy_questions/anns/qrels_test_anns_100.tsv",
                       anns_gt_path="data/researchy_questions/anns/qrels_test_anns_100.tsv")

    r_metric.evaluate_all(retrieved_dict, k=10, verbose=False)
    r_metric.evaluate_all(retrieved_dict, k=30, verbose=False)
    r_metric.evaluate_all(retrieved_dict, k=100, verbose=False)


if __name__ == '__main__':
    run_trec_evaluation(k=100, complexity=100)
