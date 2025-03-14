import logging
import os
import sys

from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    TrainerCallback
)

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import TrainDataset, TrainDatasetPreprocessed, MiniCPM_UnsupervisedDataset
from tevatron.retriever.collator import TrainCollator, TrainCollatorPreprocessed
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.trainer import TevatronTrainer as Trainer
from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer
import deepspeed
import torch
import torch.nn.functional as F

import pickle

logger = logging.getLogger(__name__)

class GradientCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.total_grad = None  
        self.step_count = 0

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs['model']
        
        grads = []
        for _, param in model.named_parameters():
                grad = deepspeed.utils.safe_get_full_grad(param)
                if grad is not None:
                    grads.append(grad.view(-1))

        grad_vector = torch.cat(grads)
        normalized_grad = F.normalize(grad_vector, p=2, dim=0)
        
        if self.total_grad is None:
            self.total_grad = torch.zeros_like(normalized_grad)
        
        self.total_grad += normalized_grad
        self.step_count += 1

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.total_grad is not None and self.step_count > 0:
                average_grad = self.total_grad / self.step_count
                normalized_final_grad = F.normalize(average_grad, p=2, dim=0).detach().cpu().float()
                
                print(f"Saving to: {args.output_dir}/normalized_final_valid_grad.pkl")
                with open(f"{args.output_dir}/normalized_final_valid_grad.pkl", "wb") as f:
                    pickle.dump(normalized_final_grad, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    
    logger.warning("This code aims to get batch level gradients. Model won't be trained or saved.")
    training_args.learning_rate = 0.0 
    training_args.lr_scheduler_type = "constant"
    training_args.optim = "sgd"
    training_args.gradient_accumulation_steps = 1
    training_args.max_grad_norm = 100000000

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    
    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )

    if "XBKYS/minicpm-embedding-data" == data_args.dataset_name:
        logger.info("Special dataset detected: MiniCPM unsupervised queries. Loading.")
        num_negs = 5
        num_pos = 1
        if data_args.train_group_size != num_negs + num_pos:
            logger.info(f"This dataset contains {num_pos} positive and {num_negs} negative per query. Setting group_size to {num_negs + num_pos}.")
            data_args.train_group_size = num_negs + num_pos
            
        train_dataset = MiniCPM_UnsupervisedDataset(data_args)
        collator = TrainCollator(data_args, tokenizer)
        train_dataset.tokenizer = tokenizer

    else:
        train_dataset = TrainDataset(data_args) if data_args.dataset_path is None else TrainDatasetPreprocessed(data_args)
        collator = TrainCollator(data_args, tokenizer) if data_args.dataset_path is None else TrainCollatorPreprocessed(data_args, tokenizer)
        train_dataset.tokenizer = tokenizer

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[GradientCallback()]
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training

if __name__ == "__main__":
    main()
