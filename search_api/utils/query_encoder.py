import torch
import time
import random
import string
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional


class QueryEncoder:
    """
    A wrapper class for encoding queries using SentenceTransformer models.
    This class handles model loading and provides an interface for encoding queries.
    """

    def __init__(
            self,
            model_name: str = "openbmb/MiniCPM-Embedding-Light",
            use_gpu: Optional[bool] = None,
            use_flash_attention: bool = False
    ):
        """
        Initialize the QueryEncoder with the specified model.

        Args:
            model_name: Name or path of the SentenceTransformer model to load
            use_gpu: Whether to use GPU for encoding. If None, will auto-detect
            use_flash_attention: Whether to use flash attention for faster inference
        """
        # Determine device based on CUDA availability if not explicitly specified
        if use_gpu is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Print device information
        print(f"Using device: {self.device}")

        # Set up model kwargs based on device and configuration
        model_kwargs = {"torch_dtype": torch.float16}

        # Add flash attention if requested and using GPU
        if use_flash_attention and self.device.type == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using flash attention for faster inference")

        # Load the model
        self.model = SentenceTransformer(model_name,
                                         trust_remote_code=True,
                                         model_kwargs=model_kwargs)

        # Move model to the appropriate device
        self.model.to(self.device)

        # Default instruction/prompt for query encoding
        self.instruction = "Instruction: Given a web search query, retrieve relevant passages that answer the query. Query: "

        print(f"Model '{model_name}' loaded successfully")

    def encode_query(self, query_text: str):
        """
        Encode a single query text into embedding.

        Args:
            query_text: The query text to encode

        Returns:
            A tensor containing the query embedding
        """
        # Convert the single string to a list for the model
        queries = [query_text]

        # Encode the query with the instruction prefix
        embedding = self.model.encode(queries, prompt=self.instruction)

        # Return the embedding tensor
        return embedding[0]

    def encode_queries(self, query_texts: List[str]):
        """
        Encode multiple query texts into embeddings.

        Args:
            query_texts: List of query texts to encode

        Returns:
            A tensor containing the query embeddings
        """
        # Encode all queries with the instruction prefix
        embeddings = self.model.encode(query_texts, prompt=self.instruction)

        # Return the embedding tensor
        return torch.tensor(embeddings, device=self.device)

    def encode_passages(self, passages: List[str]) -> torch.Tensor:
        """
        Encode passages (documents) into embeddings.
        Note: Usually passages don't need the query instruction prefix.

        Args:
            passages: List of passages to encode

        Returns:
            A tensor containing the passage embeddings
        """
        # Encode passages without the query instruction
        embeddings = self.model.encode(passages)

        # Return the embedding tensor
        return torch.tensor(embeddings, device=self.device)

    def calculate_similarity(self, query_embedding: torch.Tensor, passage_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate similarity scores between a query embedding and passage embeddings.

        Args:
            query_embedding: The query embedding tensor
            passage_embeddings: The passage embeddings tensor

        Returns:
            A tensor containing similarity scores
        """
        # Ensure query_embedding is 2D if it's just a single embedding
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        # Calculate dot product similarity
        return (query_embedding @ passage_embeddings.T).squeeze()

    def benchmark_query_time(self):
        def generate_random_text(approx_length=50):
            words = []
            current_length = 0

            # Generate words until we reach approximate desired length
            while current_length < approx_length:
                # Generate random word length between 3 and 10 characters
                word_length = random.randint(3, 10)
                word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
                words.append(word)
                current_length += word_length + 1  # +1 for space

            return ' '.join(words)

        # Generate 50 random texts
        print("Generating 50 random query texts...")
        num_queries = 50
        query_pool = [generate_random_text(50) for _ in range(num_queries)]

        # Print a few sample queries
        print(f"Sample queries from the pool:")
        for i in range(min(5, len(query_pool))):
            print(f"  {i + 1}. {query_pool[i]}")

        # Randomly select and encode 100 times
        print(f"\nRunning 100 random encoding operations...")
        total_time = 0
        encoding_times = []

        for i in range(100):
            # Randomly select a query from the pool
            random_query = random.choice(query_pool)

            # Time the encoding operation
            start_time = time.time()
            self.encode_query(random_query)
            end_time = time.time()

            # Calculate and accumulate time
            elapsed_time = end_time - start_time
            encoding_times.append(elapsed_time)
            total_time += elapsed_time

            # Progress indicator every 10 operations
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/100 encoding operations...")

        # Calculate statistics
        avg_time = total_time / 100
        min_time = min(encoding_times)
        max_time = max(encoding_times)

        # Display results
        print("\nEncoding Performance Results:")
        print(f"  Total time for 100 encoding operations: {total_time:.4f} seconds")
        print(f"  Average encoding time per query: {avg_time:.4f} seconds")
        print(f"  Minimum encoding time: {min_time:.4f} seconds")
        print(f"  Maximum encoding time: {max_time:.4f} seconds")


# Usage example
if __name__ == "__main__":
    encoder = QueryEncoder()
    encoder.benchmark_query_time()

# # Initialize the encoder
# encoder = QueryEncoder(use_flash_attention=False)
#
# # Encode a query
# query = "中国的首都是哪里？"  # "What is the capital of China?"
# query_embedding = encoder.encode_query(query)
#
# # Encode passages
# passages = ["beijing", "shanghai"]  # "北京", "上海"
# passage_embeddings = encoder.encode_passages(passages)
#
# # Calculate similarity scores
# scores = encoder.calculate_similarity(query_embedding, passage_embeddings)
#
# # Print results
# print(f"Query: {query}")
# print(f"Passages: {passages}")
# print(f"Similarity scores: {scores.tolist()}")
