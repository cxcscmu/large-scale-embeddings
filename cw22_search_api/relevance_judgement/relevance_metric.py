import csv
import json


class RMetric:
    def __init__(self, qrels_path="data/trec_eval/qrels_dev_cleaned.tsv"):
        self.qrels = self.load_qrels(qrels_path)

    def load_qrels(self, qrels_path):
        qrels = {}
        with open(qrels_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row or len(row) < 2:
                    continue
                qid = row[0].strip()
                doc_id = row[1].strip()
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(doc_id)
        return qrels

    def mrr_at_k(self, retrieved_dict, k, verbose=False):
        rank_dict = dict()
        for i in range(k + 1):
            rank_dict[i] = 0

        total_rr = 0.0
        total_queries = 0
        no_relevance_cnt = 0

        for qid, results in retrieved_dict.items():
            total_queries += 1
            rr = 0.0
            relevant_docs = self.qrels.get(qid, set())
            if not relevant_docs:
                no_relevance_cnt += 1

            for rank, result in enumerate(results, start=1):
                try:
                    # Decode byte string to a UTF-8 string and load JSON object
                    result_str = result.decode('utf-8')
                    doc_info = json.loads(result_str)
                except Exception as e:
                    print(f"Error processing result for query {qid} at rank {rank}: {e}")
                    continue

                url = doc_info.get("URL", "").strip()
                url_hash = doc_info.get("URL-hash", "").strip()
                doc_id = doc_info.get("ClueWeb22-ID", "").strip()

                if verbose:
                    print("url, url_hash, CW22_doc_id", url, url_hash, doc_id)
                    print("relevant_docs", relevant_docs)

                if doc_id in relevant_docs or url in relevant_docs or url_hash in relevant_docs:
                    rank_dict[0] += 1
                    rank_dict[rank] += 1
                    rr = 1.0 / rank
                    break
            total_rr += rr

        print("rank_dict:", rank_dict)
        print("Query without relevance:", no_relevance_cnt, "out of total_queries:", total_queries)
        raw_value = total_rr / total_queries if total_queries > 0 else 0.0
        print(f"MMR@{k}: {raw_value}")

        total_queries -= no_relevance_cnt
        corrected_value = total_rr / total_queries if total_queries > 0 else 0.0
        print(f"MMR@{k} corrected: {corrected_value}")
