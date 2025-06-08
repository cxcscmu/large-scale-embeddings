import csv
import json
import math
import numpy as np
from collections import defaultdict


class RMetric:
    def __init__(self, qrels_path="data/trec_eval/qrels_dev_cleaned.tsv", anns_gt_path=None):
        self.qrels = self.load_qrels(qrels_path)
        self.anns_gt = self.load_anns_ground_truth(anns_gt_path) if anns_gt_path else None

    def load_qrels(self, qrels_path):
        """
        Load query relevance judgments from TSV file
        Format: qid, doc_id
        Returns a dictionary mapping query IDs to sets of relevant document IDs
        """
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
        
    def load_anns_ground_truth(self, anns_gt_path):
        """
        Load ANNS ground truth from TSV file
        Format: qid, doc_id
        The first k entries for each qid are considered the ground truth for ANNS
        Returns a dictionary mapping query IDs to ordered lists of document IDs
        """
        if not anns_gt_path:
            return None
            
        anns_gt = defaultdict(list)
        try:
            with open(anns_gt_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    qid = row[0].strip()
                    doc_id = row[1].strip()
                    # Maintain the order of docs for each query
                    anns_gt[qid].append(doc_id)
            return dict(anns_gt)
        except Exception as e:
            print(f"Error loading ANNS ground truth: {e}")
            return None

    def _is_relevant(self, doc_info, relevant_docs):
        """
        Check if a document is relevant based on different possible IDs
        """
        url = doc_info.get("URL", "").strip()
        url_hash = doc_info.get("URL-hash", "").strip()
        doc_id = doc_info.get("ClueWeb22-ID", "").strip()
        
        return doc_id in relevant_docs or url in relevant_docs or url_hash in relevant_docs

    def _parse_result(self, result):
        """
        Parse a result string into a document info dictionary
        """
        try:
            # Decode byte string to a UTF-8 string and load JSON object
            result_str = result.decode('utf-8')
            doc_info = json.loads(result_str)
            return doc_info
        except Exception as e:
            print(f"Error processing result: {e}")
            return {}

    def mrr_at_k(self, retrieved_dict, k, verbose=False):
        """
        Calculate Mean Reciprocal Rank at k
        MRR measures the reciprocal of the rank at which the first relevant document is found
        """
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

            for rank, result in enumerate(results[:k], start=1):
                doc_info = self._parse_result(result)
                if not doc_info:
                    continue

                if verbose:
                    url = doc_info.get("URL", "").strip()
                    url_hash = doc_info.get("URL-hash", "").strip()
                    doc_id = doc_info.get("ClueWeb22-ID", "").strip()
                    print("url, url_hash, CW22_doc_id", url, url_hash, doc_id)
                    print("relevant_docs", relevant_docs)

                if self._is_relevant(doc_info, relevant_docs):
                    rank_dict[0] += 1
                    rank_dict[rank] += 1
                    rr = 1.0 / rank
                    break
            total_rr += rr

        print("rank_dict:", rank_dict)
        print("Query without relevance:", no_relevance_cnt, "out of total_queries:", total_queries)
        raw_value = total_rr / total_queries if total_queries > 0 else 0.0
        print(f"MRR@{k}: {raw_value}")
        return raw_value

    def precision_at_k(self, retrieved_dict, k, verbose=False):
        """
        Calculate Precision at k
        Precision@k is the proportion of retrieved documents that are relevant
        """
        total_precision = 0.0
        total_queries = 0
        
        for qid, results in retrieved_dict.items():
            total_queries += 1
            relevant_docs = self.qrels.get(qid, set())
            
            relevant_count = 0
            for rank, result in enumerate(results[:k], start=1):
                doc_info = self._parse_result(result)
                if not doc_info:
                    continue
                
                if self._is_relevant(doc_info, relevant_docs):
                    relevant_count += 1
            
            # Precision@k for this query
            precision = relevant_count / min(k, len(results)) if results else 0
            if verbose:
                print(f"Query {qid}: Precision@{k} = {precision} ({relevant_count}/{min(k, len(results))})")
            
            total_precision += precision
        
        avg_precision = total_precision / total_queries if total_queries > 0 else 0.0
        print(f"Precision@{k}: {avg_precision}")
        return avg_precision

    def recall_at_k(self, retrieved_dict, k, verbose=False):
        """
        Calculate Recall at k
        Recall@k is the proportion of relevant documents that are retrieved within the top k results
        """
        total_recall = 0.0
        total_queries = 0
        
        for qid, results in retrieved_dict.items():
            total_queries += 1
            relevant_docs = self.qrels.get(qid, set())
            if not relevant_docs:
                continue
            
            relevant_retrieved = 0
            for rank, result in enumerate(results[:k], start=1):
                doc_info = self._parse_result(result)
                if not doc_info:
                    continue
                
                if self._is_relevant(doc_info, relevant_docs):
                    relevant_retrieved += 1
            
            # Recall@k for this query
            recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
            if verbose:
                print(f"Query {qid}: Recall@{k} = {recall} ({relevant_retrieved}/{len(relevant_docs)})")
            
            total_recall += recall
        
        avg_recall = total_recall / total_queries if total_queries > 0 else 0.0
        print(f"Recall@{k}: {avg_recall}")
        return avg_recall

    def ap_at_k(self, qid, results, k, verbose=False):
        """
        Calculate Average Precision at k for a single query
        AP is the average of precision values at each relevant document position
        """
        relevant_docs = self.qrels.get(qid, set())
        if not relevant_docs:
            return 0.0
        
        relevant_count = 0
        sum_precision = 0.0
        
        for rank, result in enumerate(results[:k], start=1):
            doc_info = self._parse_result(result)
            if not doc_info:
                continue
            
            if self._is_relevant(doc_info, relevant_docs):
                relevant_count += 1
                # Precision at this position
                precision_at_rank = relevant_count / rank
                sum_precision += precision_at_rank
                
                if verbose:
                    print(f"Query {qid}, Rank {rank}: Relevant, Precision = {precision_at_rank}")
        
        # AP is the sum of precision at relevant positions divided by the total number of relevant documents
        ap = sum_precision / len(relevant_docs) if relevant_docs else 0.0
        
        if verbose:
            print(f"Query {qid}: AP@{k} = {ap}")
        
        return ap

    def map_at_k(self, retrieved_dict, k, verbose=False):
        """
        Calculate Mean Average Precision at k
        MAP is the mean of the average precision scores for each query
        """
        total_ap = 0.0
        total_queries = 0
        
        for qid, results in retrieved_dict.items():
            total_queries += 1
            ap = self.ap_at_k(qid, results, k, verbose)
            total_ap += ap
        
        map_score = total_ap / total_queries if total_queries > 0 else 0.0
        print(f"MAP@{k}: {map_score}")
        return map_score

    def dcg_at_k(self, qid, results, k, verbose=False):
        """
        Calculate Discounted Cumulative Gain at k for a single query
        DCG measures the quality of ranking by considering both relevance and position
        Uses binary relevance (1 for relevant, 0 for not relevant)
        """
        relevant_docs = self.qrels.get(qid, set())
        if not relevant_docs:
            return 0.0
        
        dcg = 0.0
        
        for rank, result in enumerate(results[:k], start=1):
            doc_info = self._parse_result(result)
            if not doc_info:
                continue
            
            # Binary relevance: 1 if relevant, 0 if not
            rel = 1 if self._is_relevant(doc_info, relevant_docs) else 0
            
            # DCG formula with binary relevance: rel_i / log2(i+1)
            dcg += rel / math.log2(rank + 1)
            
            if verbose and rel > 0:
                print(f"Query {qid}, Rank {rank}: Relevant, DCG contribution = {rel / math.log2(rank + 1)}")
        
        if verbose:
            print(f"Query {qid}: DCG@{k} = {dcg}")
        
        return dcg

    def ideal_dcg_at_k(self, qid, k, verbose=False):
        """
        Calculate Ideal DCG at k for a single query
        IDCG is the DCG value for the ideal ranking (all relevant docs at the top)
        """
        relevant_docs = self.qrels.get(qid, set())
        if not relevant_docs:
            return 0.0
        
        # For binary relevance, IDCG is the sum of discounted gains for optimal ranking
        # where all relevant docs are at the top
        idcg = 0.0
        num_rel = min(len(relevant_docs), k)
        
        for i in range(1, num_rel + 1):
            idcg += 1.0 / math.log2(i + 1)
        
        if verbose:
            print(f"Query {qid}: IDCG@{k} = {idcg} (for {num_rel} relevant docs)")
        
        return idcg

    def ndcg_at_k(self, retrieved_dict, k, verbose=False):
        """
        Calculate Normalized Discounted Cumulative Gain at k
        NDCG normalizes DCG by the ideal DCG to give a score between 0 and 1
        """
        total_ndcg = 0.0
        total_queries = 0
        
        for qid, results in retrieved_dict.items():
            total_queries += 1
            
            dcg = self.dcg_at_k(qid, results, k, verbose)
            idcg = self.ideal_dcg_at_k(qid, k, verbose)
            
            # NDCG is DCG / IDCG (if IDCG is 0, NDCG is 0)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            if verbose:
                print(f"Query {qid}: NDCG@{k} = {ndcg} (DCG={dcg}, IDCG={idcg})")
            
            total_ndcg += ndcg
        
        avg_ndcg = total_ndcg / total_queries if total_queries > 0 else 0.0
        print(f"NDCG@{k}: {avg_ndcg}")
        return avg_ndcg
    
    def anns_recall_at_k(self, retrieved_dict, k, anns_k=None, verbose=False):
        """
        Calculate ANNS Recall@k
        Measures overlap between retrieved documents and ANNS ground truth
        
        Args:
            retrieved_dict: Dictionary mapping query IDs to retrieved documents
            k: Number of top retrieved results to consider
            anns_k: Number of ground truth ANNS documents to consider (default: same as k)
            verbose: Whether to print detailed information
            
        Returns:
            Average ANNS recall across all queries
        """
        if not self.anns_gt:
            print("Error: ANNS ground truth not loaded. Initialize with anns_gt_path parameter.")
            return 0.0
        
        if anns_k is None:
            anns_k = k
            
        total_recall = 0.0
        total_queries = 0
        
        for qid, results in retrieved_dict.items():
            if qid not in self.anns_gt:
                if verbose:
                    print(f"Query {qid} not found in ANNS ground truth, skipping.")
                continue
                
            total_queries += 1
            anns_docs = self.anns_gt[qid][:anns_k]  # Take top-k docs from ground truth
            
            if not anns_docs:
                if verbose:
                    print(f"No ANNS ground truth for query {qid}, skipping.")
                continue
                
            # Extract the retrieved document IDs
            retrieved_docs = []
            for result in results[:k]:
                doc_info = self._parse_result(result)
                if not doc_info:
                    continue
                    
                doc_id = doc_info.get("ClueWeb22-ID", "").strip()
                url = doc_info.get("URL", "").strip()
                url_hash = doc_info.get("URL-hash", "").strip()
                
                # Add all possible identifiers to the retrieved docs list
                if doc_id:
                    retrieved_docs.append(doc_id)
                if url:
                    retrieved_docs.append(url)
                if url_hash:
                    retrieved_docs.append(url_hash)
            
            # Calculate the intersection size
            intersection = set(retrieved_docs).intersection(set(anns_docs))
            
            # Calculate recall for this query
            recall = len(intersection) / len(anns_docs)
            
            if verbose:
                print(f"Query {qid}:")
                print(f"  - ANNS GT [{anns_k}]: {anns_docs}")
                print(f"  - Retrieved [{k}]: {retrieved_docs}")
                print(f"  - Intersection: {intersection}")
                print(f"  - ANNS Recall@{k}: {recall} ({len(intersection)}/{len(anns_docs)})")
            
            total_recall += recall
        
        avg_recall = total_recall / total_queries if total_queries > 0 else 0.0
        print(f"ANNS Recall@{k}: {avg_recall}")
        return avg_recall

    def evaluate_all(self, retrieved_dict, k, verbose=False):
        """
        Run all evaluation metrics at k and return results
        """
        results = {}
        
        print(f"\nEvaluating at k={k}:")
        print("-" * 50)
        
        results["MRR"] = self.mrr_at_k(retrieved_dict, k, verbose)
        results["Precision"] = self.precision_at_k(retrieved_dict, k, verbose)
        results["Recall"] = self.recall_at_k(retrieved_dict, k, verbose)
        results["MAP"] = self.map_at_k(retrieved_dict, k, verbose)
        results["NDCG"] = self.ndcg_at_k(retrieved_dict, k, verbose)
        
        # Only run ANNS recall if ground truth is available
        if self.anns_gt:
            results["ANNS_Recall"] = self.anns_recall_at_k(retrieved_dict, k, verbose=verbose)
        
        print("-" * 50)
        return results