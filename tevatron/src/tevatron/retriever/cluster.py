import faiss
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.metrics import silhouette_score
import torch
import torch.nn.functional as F


import torch
import torch.nn.functional as F


class Silhouette:
    @staticmethod
    def score(X, labels, loss=False, batch_size=1000):
        """Compute the mean Silhouette Coefficient of all samples using cosine distance.

        Parameters:
        X : tensor [n_samples, n_features]
            Feature array.
        labels : tensor [n_samples]
            Label values for each sample.
        loss : bool
            If True, will return the negative silhouette score (useful for backpropagation).
        batch_size : int
            The batch size for processing the cosine distance computation in chunks.

        Returns:
        silhouette : float
            Mean Silhouette Coefficient for all samples.
        """

        X = torch.tensor(X).cuda()
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long, device=X.device)

        unique_labels = torch.unique(labels)

        # Compute intra-cluster distances
        A = Silhouette._intra_cluster_distances(X, labels, unique_labels, batch_size)

        # Compute nearest-cluster distances
        B = Silhouette._nearest_cluster_distances(X, labels, unique_labels, batch_size)

        # Calculate silhouette coefficient for each sample
        sil_samples = (B - A) / torch.maximum(A, B)

        # Handle NaN values (clusters of size 1)
        sil_samples = torch.nan_to_num(sil_samples, nan=0.0)

        # Return mean silhouette score
        mean_sil_score = torch.mean(sil_samples)
        return -mean_sil_score if loss else mean_sil_score.item()

    @staticmethod
    def _intra_cluster_distances(X, labels, unique_labels, batch_size):
        """Compute the mean intra-cluster distance for each sample using cosine distance."""
        intra_dist = torch.zeros(labels.size(), dtype=torch.float32, device=X.device)
        for label in unique_labels:
            # Find indices of samples belonging to the same cluster
            cluster_indices = torch.where(labels == label)[0]
            subX = X[cluster_indices]

            # Compute pairwise cosine distances within the cluster in batches
            distances = Silhouette._cosine_distance_matrix_batch(subX, subX, batch_size)
            intra_dist[cluster_indices] = distances.sum(dim=1) / (distances.size(0) - 1)

        return intra_dist

    @staticmethod
    def _nearest_cluster_distances(X, labels, unique_labels, batch_size):
        """Compute the mean nearest-cluster distance for each sample using cosine distance."""
        inter_dist = torch.full(
            labels.size(), torch.inf, dtype=torch.float32, device=X.device
        )

        # Compute the pairwise distance between each cluster
        for label_a, label_b in torch.combinations(unique_labels, r=2):
            cluster_a_indices = torch.where(labels == label_a)[0]
            cluster_b_indices = torch.where(labels == label_b)[0]

            subX_a = X[cluster_a_indices]
            subX_b = X[cluster_b_indices]

            # Compute distances between all points in cluster_a and all points in cluster_b in batches
            dist = Silhouette._cosine_distance_matrix_batch(subX_a, subX_b, batch_size)
            dist_a = dist.mean(dim=1)
            dist_b = dist.mean(dim=0)

            # Update nearest-cluster distance for samples in cluster_a and cluster_b
            inter_dist[cluster_a_indices] = torch.minimum(
                inter_dist[cluster_a_indices], dist_a
            )
            inter_dist[cluster_b_indices] = torch.minimum(
                inter_dist[cluster_b_indices], dist_b
            )

        return inter_dist

    @staticmethod
    def _cosine_distance_matrix_batch(A, B, batch_size):
        """Compute cosine distances between A and B in batches to save memory."""
        num_samples_a, num_samples_b = A.size(0), B.size(0)
        cosine_dist = torch.zeros(
            (num_samples_a, num_samples_b), dtype=torch.float32, device=A.device
        )

        for i in range(0, num_samples_a, batch_size):
            end_i = min(i + batch_size, num_samples_a)
            for j in range(0, num_samples_b, batch_size):
                end_j = min(j + batch_size, num_samples_b)

                # Get the current batch of vectors
                A_batch = A[i:end_i]
                B_batch = B[j:end_j]

                # Normalize the vectors
                A_norm = F.normalize(A_batch, p=2, dim=1)
                B_norm = F.normalize(B_batch, p=2, dim=1)

                # Compute cosine similarity and convert to distance
                cosine_sim = torch.mm(A_norm, B_norm.t())
                cosine_dist[i:end_i, j:end_j] = 1 - cosine_sim

        return cosine_dist


class KMeansTuner:
    def __init__(self, gpu: bool = True):
        """
        Initialize KMeans tuner using silhouette score.

        Args:
            gpu: Whether to use GPU acceleration
        """
        self.use_gpu = gpu

    def compute_silhouette(self, vectors: np.ndarray, k: int) -> float:
        """
        Compute silhouette score for a given k.

        Args:
            vectors: Input vectors (n_samples, n_features)
            k: Number of clusters

        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        vectors = vectors.astype(np.float32)

        kmeans = faiss.Kmeans(
            d=vectors.shape[1],
            k=k,
            niter=20,
            gpu=self.use_gpu,
            spherical=True,
        )

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            gpu_index = faiss.GpuIndexFlatL2(
                res, vectors.shape[1], cfg
            )  # GpuIndexFlatL2 ?
            kmeans.index = gpu_index

        kmeans.train(vectors)

        assignments = kmeans.assign(vectors)[1]
        assignments = assignments.flatten()

        if k <= 1 or k >= vectors.shape[0]:
            return -1.0

        try:
            score = Silhouette.score(vectors, assignments)  # (, metric="cosine")
            return float(score)
        except ValueError:
            return -1.0

    def find_optimal_k(
        self, vectors: np.ndarray, k_range: List[int], plot: bool = True
    ) -> Tuple[int, List[float]]:
        """
        Find optimal k using silhouette analysis.

        Args:
            vectors: Input vectors
            k_range: List of k values to try
            plot: Whether to plot the silhouette scores

        Returns:
            Tuple of (optimal k, list of silhouette scores for each k)
        """
        print("Computing silhouette scores for different k values...")
        silhouette_scores = []

        for k in k_range:
            print(f"Testing k={k}")
            score = self.compute_silhouette(vectors, k)
            silhouette_scores.append(score)

        # Find k with highest silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, silhouette_scores, "b-", marker="o")
            plt.axvline(
                x=optimal_k, color="r", linestyle="--", label=f"Optimal k={optimal_k}"
            )
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Analysis for Optimal k")
            plt.legend()
            plt.grid(True)
            plt.savefig("silhouette_analysis_plot.pdf", format="pdf")
            plt.close()

        return optimal_k, silhouette_scores

    @staticmethod
    def interpret_scores(scores: List[float]) -> str:
        """
        Provide interpretation of silhouette scores.

        Args:
            scores: List of silhouette scores

        Returns:
            String with interpretation
        """
        max_score = max(scores)
        if max_score < 0:
            return "Poor clustering structure: Consider data preprocessing or different algorithm"
        elif max_score < 0.25:
            return "Weak clustering structure: Clusters are not well-separated"
        elif max_score < 0.5:
            return "Medium clustering structure: Clusters are moderately separated"
        elif max_score < 0.75:
            return "Strong clustering structure: Well-separated clusters"
        else:
            return "Very strong clustering structure: Highly separated clusters"


class SimpleKMeans:
    def __init__(self, n_clusters: int, use_gpu: bool):
        """
        Initialize simple KMeans clustering.

        Args:
            n_clusters: Number of clusters to create
        """
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu
        self.clusters = None  # Will store list of document indices for each cluster

    def fit(self, vectors: np.ndarray) -> List[np.ndarray]:
        """
        Cluster the input vectors and return document indices for each cluster.

        Args:
            vectors: Input vectors of shape (n_docs, dim)

        Returns:
            List where each element contains indices of documents in that cluster
        """
        kmeans = faiss.Kmeans(
            d=vectors.shape[1],
            k=self.n_clusters,
            niter=20,
            gpu=self.use_gpu,
            spherical=True,
        )

        vectors = vectors.astype(np.float32)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            gpu_index = faiss.GpuIndexFlatL2(
                res, vectors.shape[1], cfg
            )  # GpuIndexFlatL2 ?
            kmeans.index = gpu_index

        kmeans.train(vectors)

        assignments = kmeans.assign(vectors)[1]

        # Group document indices by cluster
        self.clusters = [[] for _ in range(self.n_clusters)]
        for doc_idx, cluster_id in enumerate(assignments):
            self.clusters[cluster_id].append(doc_idx)

        # Convert to list of lists for easier handling
        self.clusters = [sorted(cluster) for cluster in self.clusters]

        return self.clusters

    def get_cluster(self, cluster_id: int) -> np.ndarray:
        """
        Get document indices for a specific cluster.

        Args:
            cluster_id: ID of the cluster (0 to n_clusters-1)

        Returns:
            Array of document indices belonging to this cluster
        """
        if self.clusters is None:
            raise ValueError("Must call fit() before getting clusters")
        return self.clusters[cluster_id]


import pickle


def pickle_load(path):
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


if __name__ == "__main__":
    import json

    # Load your vectors
    path_to_vecs = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-finetune-ep1-1/query-marco-train.pkl"
    q_reps, q_lookup = pickle_load(path_to_vecs)

    print(f"Loaded {len(q_reps)} vectors of dimension {q_reps.shape[1]}")
    print(f"Vector norms: {np.linalg.norm(q_reps[0])}")

    tuner = KMeansTuner(gpu=True)

    # Define range of k values to test
    k_range = [2, 5, 10, 20, 50, 100, 200, 500]

    # Find optimal k
    optimal_k, objectives = tuner.find_optimal_k(q_reps, k_range, plot=True)

    print(f"Optimal number of clusters: {optimal_k}")

    # query_dict = {}

    # # Read and parse each line
    # with open(
    #     "/data/user_data/jmcoelho/datasets/marco/documents/train.query.jsonl", "r"
    # ) as file:
    #     for line in file:
    #         data = json.loads(line)
    #         query_dict[int(data["query_id"])] = data["query"]

    # # Initialize KMeans
    # n_clusters = 100  # you can adjust this number
    # kmeans = SimpleKMeans(n_clusters=n_clusters, use_gpu=True)

    # # Fit and get clusters
    # print("Starting clustering...")
    # clusters = kmeans.fit(q_reps)

    # # Print some statistics
    # print("\nClustering Results:")
    # print(f"Number of clusters: {len(clusters)}")

    # # Print size of first 5 clusters
    # print("\nFirst 5 clusters sizes:")
    # for i in range(min(5, len(clusters))):
    #     print(f"Cluster {i}: {len(clusters[i])} documents")

    # # Print some example documents from first cluster
    # print("\nExample documents from first cluster:")
    # first_cluster = clusters[0]
    # for idx in first_cluster[:5]:  # first 5 documents in first cluster
    #     print(
    #         f"Document {idx}: {query_dict[int(q_lookup[idx])] if q_lookup else 'No lookup available'}"
    #    )
