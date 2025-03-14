import torch
import logging
from tevatron.retriever.modeling import DenseModel, EncoderOutput
from typing import Dict

logger = logging.getLogger(__name__)


class DenseModelWithNegativeCache(DenseModel):

    def init_cache(self, reps):
        self.negative_cache = reps.detach()

        logger.info(f"Initialized cache. Shape: {self.negative_cache.shape}")

    def forward(
        self,
        query: Dict[str, torch.Tensor] = None,
        passage: Dict[str, torch.Tensor] = None,
        use_cache: bool = True,
    ):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference - ddp instance not supported - model must fit in a single gpu...
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)

        if self.negative_cache is not None and use_cache:
            p_reps = torch.cat((p_reps, self.negative_cache), dim=0)

        scores = self.compute_similarity(q_reps, p_reps)
        scores = scores.view(q_reps.size(0), -1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (p_reps.size(0) // q_reps.size(0))

        loss = self.compute_loss(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size  # counter average weight reduction

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
