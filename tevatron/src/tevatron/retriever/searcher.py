import faiss
import numpy as np
from tqdm import tqdm
import torch
import logging

logger = logging.getLogger(__name__)


class FaissFlatSearcher:
    def __init__(self, init_reps: np.ndarray):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        self.index = index

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def move_index_to_gpu(self):
        try:
            num_gpus = torch.cuda.device_count()
        except Exception:
            logger.warning("Unable to assess number of GPU devices. Will conduct CPU search.")
            return

        if num_gpus == 0:
            logger.warning("No GPUs available. Will conduct CPU search.")
            return

        elif num_gpus == 1:
            logger.info(f"Detected 1 GPU. Moving index to single GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        else:
            logger.info(
                f"Detected {num_gpus} GPUs. Sharing index across GPUs."
            )
            gpu_resources = [faiss.StandardGpuResources() for _ in range(num_gpus)]

            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.verbose = True

            self.index = faiss.index_cpu_to_gpu_multiple_py(
                gpu_resources,
                self.index,
                co,
            )

    def batch_search(
        self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool = False
    ):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(
                q_reps[start_idx : start_idx + batch_size], k
            )
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


class FaissSearcher(FaissFlatSearcher):

    def __init__(self, init_reps: np.ndarray, factory_str: str):
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        self.index = index
        self.index.verbose = True
        if not self.index.is_trained:
            self.index.train(init_reps)
