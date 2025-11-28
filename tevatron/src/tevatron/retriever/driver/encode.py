import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron.retriever.arguments import ModelArguments, \
    TevatronDataArguments as DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import EncodeDataset, EncodeDataset_MARCOWeb, EncodeDataset_ClueWeb22, EncodeDataset_Amazon
from tevatron.retriever.collator import EncodeCollator
from tevatron.retriever.modeling import EncoderOutput, DenseModel

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    model = DenseModel.load(
        model_args.model_name_or_path,
        pooling=model_args.pooling,
        normalize=model_args.normalize,
        lora_name_or_path=model_args.lora_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype
    )

    if data_args.clueweb_api_dataset: 
        dataset_obj = EncodeDataset_ClueWeb22
    elif data_args.marcoweb_api_dataset: 
        dataset_obj = EncodeDataset_MARCOWeb
    elif data_args.amazon_dataset:
        dataset_obj = EncodeDataset_Amazon
    else:
        dataset_obj = EncodeDataset
        
    encode_dataset = dataset_obj(
        data_args=data_args,
    )

    encode_collator = EncodeCollator(
        data_args=data_args,
        tokenizer=tokenizer,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    dtype = None
    if training_args.fp16:
        print("Set encoding precision: fp16")
        dtype = torch.float16
    elif training_args.bf16:
        print("Set encoding precision: bf16")
        dtype = torch.bfloat16

    # preemption support 
    batch_output_path = data_args.encode_output_path + ".temp_emb"
    batch_indices_output_path = data_args.encode_output_path + ".temp_idx"
    if os.path.exists(batch_output_path) and os.path.exists(batch_indices_output_path): 
        with open(batch_indices_output_path, 'rb') as f:
            lookup_indices = pickle.load(f) 
        with open(batch_output_path, 'rb') as f:
            encoded = pickle.load(f) 
        starting_id = lookup_indices[-1]
        start = False 
    else: 
        start = True 

    print("Starting to encode.")
    for (batch_ids, batch) in tqdm(encode_loader):

        # skip completed batches 
        if not start: 
            # continue from next batch  
            if starting_id == batch_ids[-1]: 
                start = True
            continue

        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast(dtype=dtype) if dtype is not None else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_query:
                    model_output: EncoderOutput = model(query=batch)
                    encoded.append(model_output.q_reps.cpu().detach().numpy())
                else:
                    model_output: EncoderOutput = model(passage=batch)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())

        # preemption support: save per 50 batches 
        if (len(lookup_indices) // len(batch_ids)) % training_args.inference_save_step == 0: 
            with open(batch_output_path, 'wb') as f:
                pickle.dump(encoded, f)
            with open(batch_indices_output_path, 'wb') as f:
                pickle.dump(lookup_indices, f)
            logger.info("saving checkpoint...")


    # check if all data elements for this shard present 
    assert len(lookup_indices) == len(encode_dataset)
    for i in range(len(encode_dataset)): 
        assert encode_dataset[i][0] == lookup_indices[i], f"The {i}th element from lookup indices is different from the corresponding element in the datasets"

    encoded = np.concatenate(encoded)

    with open(data_args.encode_output_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)

    # remove the preemption files 
    if os.path.exists(batch_indices_output_path):
        os.remove(batch_indices_output_path)  
    if os.path.exists(batch_output_path): 
        os.remove(batch_output_path)


if __name__ == "__main__":
    main()
