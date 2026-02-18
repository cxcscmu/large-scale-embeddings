import json
import threading
from typing import List, Optional, Tuple

import torch
from sentence_transformers import CrossEncoder


class DocReranker:
    """
    A reranker wrapper for CrossEncoder models (e.g., openbmb/MiniCPM-Reranker-Light).
    Takes a list of raw document byte strings and a query, returns them reordered by relevance.
    """

    MAX_DOCS = 100  # Skip reranking if doc count exceeds this threshold

    def __init__(
            self,
            model_name: str = "openbmb/MiniCPM-Reranker-Light",
            max_length: int = 1024,
    ):
        """
        Initialize the DocReranker.

        Args:
            model_name: HuggingFace model name or local path
            max_length: Maximum token length for query+doc pairs
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            trust_remote_code=True,
            device=self.device,
            automodel_args={"torch_dtype": torch.float16, "attn_implementation": "eager"},
        )
        self.model.tokenizer.padding_side = "right"

        # Ensure the underlying HuggingFace model is on the correct device.
        # CrossEncoder may initialize on CPU internally even when device is passed.
        self.model.model.to(self.device)

        self.instruction = "Query: "
        self._lock = threading.Lock()

        print(f"DocReranker: model '{model_name}' loaded on {self.device}")

    def _extract_clean_text(self, doc_bytes: bytes) -> str:
        """
        Decode a document byte string and extract its Clean-Text field.

        Args:
            doc_bytes: Raw bytes of a JSON-encoded document

        Returns:
            Clean-Text string, or empty string if extraction fails
        """
        try:
            doc = json.loads(doc_bytes.decode("utf-8"))
            return doc.get("Clean-Text", "")
        except Exception:
            return ""

    def _predict(self, sentence_pairs: List[List[str]]) -> List[float]:
        """Run CrossEncoder inference under lock to ensure thread safety."""
        with self._lock:
            return self.model.predict(
                sentence_pairs,
                convert_to_tensor=False,
                show_progress_bar=False,
            )

    def rerank(self, doc_texts: List[bytes], query: str) -> List[bytes]:
        """
        Rerank a list of document byte strings by relevance to the query.
        If len(doc_texts) > MAX_DOCS, returns the list unchanged.

        Args:
            doc_texts: List of JSON-encoded document byte strings
            query: The search query string

        Returns:
            Reordered list of document byte strings (most relevant first)
        """
        if not doc_texts or len(doc_texts) > self.MAX_DOCS:
            return doc_texts

        prefixed_query = self.instruction + query
        passages = [self._extract_clean_text(doc) for doc in doc_texts]
        sentence_pairs = [[prefixed_query, p] for p in passages]

        scores = self._predict(sentence_pairs)

        ranked = sorted(zip(scores, doc_texts), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]

    def rerank_with_indices(self, doc_texts: List[bytes], query: str) -> Tuple[List[bytes], List[int]]:
        """
        Rerank documents and also return the sorted original indices.
        Use this when you need to reorder parallel arrays (e.g. distances, docids).

        Args:
            doc_texts: List of JSON-encoded document byte strings
            query: The search query string

        Returns:
            (sorted_doc_texts, sorted_indices) — sorted_indices[i] is the original
            index of the document now at position i.
        """
        if not doc_texts or len(doc_texts) > self.MAX_DOCS:
            return doc_texts, list(range(len(doc_texts)))

        prefixed_query = self.instruction + query
        passages = [self._extract_clean_text(doc) for doc in doc_texts]
        sentence_pairs = [[prefixed_query, p] for p in passages]

        scores = self._predict(sentence_pairs)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        sorted_indices = [i for i, _ in ranked]
        sorted_doc_texts = [doc_texts[i] for i in sorted_indices]
        return sorted_doc_texts, sorted_indices


if __name__ == "__main__":
    # Minimal test using the official MiniCPM-Reranker-Light example
    reranker = DocReranker()

    query = "Where is the capital of China?"
    passages = ["beijing", "shanghai"]

    # --- Test 1: raw CrossEncoder API (mirrors official example) ---
    print("=== Test 1: Raw CrossEncoder API ===")
    prefixed_query = reranker.instruction + query
    sentence_pairs = [[prefixed_query, doc] for doc in passages]

    with reranker._lock:
        scores = reranker.model.predict(sentence_pairs, convert_to_tensor=True).tolist()
        rankings = reranker.model.rank(query, passages, return_documents=True, convert_to_tensor=True)

    print("Scores:", scores)
    for r in rankings:
        print(f"Score: {r['score']:.4f}, Corpus: {r['text']}")

    # --- Test 2: rerank() with byte-string documents ---
    print("\n=== Test 2: rerank() with JSON byte documents ===")
    doc_bytes_list = [
        json.dumps({"Clean-Text": p, "URL": f"http://example.com/{i}"}).encode("utf-8")
        for i, p in enumerate(passages)
    ]

    reranked = reranker.rerank(doc_bytes_list, query)
    for doc in reranked:
        obj = json.loads(doc.decode("utf-8"))
        print(f"  -> {obj['Clean-Text']}")

    # --- Test 3: rerank_with_indices() ---
    print("\n=== Test 3: rerank_with_indices() ===")
    reranked_docs, order = reranker.rerank_with_indices(doc_bytes_list, query)
    print("Sorted original indices:", order)
    for idx, doc in zip(order, reranked_docs):
        obj = json.loads(doc.decode("utf-8"))
        print(f"  original[{idx}] -> {obj['Clean-Text']}")
