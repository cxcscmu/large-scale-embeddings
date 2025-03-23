import os
import random
from typing import List, Tuple
import pandas as pd

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

from huggingface_hub import login

from tevatron.retriever.arguments import DataArguments
from tevatron.utils.ClueWeb22Api import ClueWeb22Api, create_shards

from transformers import BatchEncoding

import logging

logger = logging.getLogger(__name__)


def format_query(query: str, prefix: str = "") -> str:
    return f"{prefix} {query.strip()}".strip()


def format_passage(
    text: str, title: str = "", prefix: str = "", add_markers: bool = False
) -> str:
    if add_markers:
        return f"{prefix} Title: {title.strip()} Text: {text.strip()}".strip()
    else:
        return f"{prefix} {title.strip()} {text.strip()}".strip()


class Cropping:
    def __init__(self, ratio_min: float = 0.1, ratio_max: float = 0.5):
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max

    def augment(self, data: str) -> str:
        words = data.split()
        ratio = random.uniform(self.ratio_min, self.ratio_max)
        length = int(len(words) * ratio)
        start = random.randint(0, len(words) - length)
        end = start + length
        cropped_words = words[start:end]
        return " ".join(cropped_words)

    def __call__(self, data: List[int]) -> List[int]:
        return self.augment(data)


class PretrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer=None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

        self.strategy = Cropping()

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]

        content = f"{group['title']} {group['text']}"  # assumes dataset is a jsonl with title and text fields

        formated_query = self.strategy(content)
        formated_passages = [self.strategy(content)]

        return formated_query, formated_passages


class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer=None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group["query"]
        group_positives = group["positive_passages"]
        group_negatives = group["negative_passages"]

        formated_query = format_query(query, self.data_args.query_prefix)
        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        formated_passages.append(
            format_passage(
                pos_psg["text"], pos_psg["title"], self.data_args.passage_prefix
            )
        )

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset : _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(
                format_passage(
                    neg_psg["text"], neg_psg["title"], self.data_args.passage_prefix
                )
            )

        return formated_query, formated_passages


class MiniCPM_UnsupervisedDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer=None):
        self.data_args = data_args

        # data_files = [f"en_{str(i).zfill(2)}.jsonl" for i in range(24)]
        # self.train_data = load_dataset("XBKYS/minicpm-embedding-data", data_files=data_files, split="train")
        # # self.train_data = load_dataset("XBKYS/minicpm-embedding-data", data_files={
        # #         'train': [f'en_{i:02d}.jsonl' for i in range(24)]
        # # }, cache_dir=self.data_args.dataset_cache_dir)

        # self.train_data = self.train_data.shuffle(seed=42).select(range(180000))

        print("HARDCODED: Loading a filtered version from disk.")
        self.train_data = load_from_disk(
            "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/mates_15q_6d_multiple_valid"
        )

        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        try:
            return len(self.train_data["train"])
        except Exception:
            return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        try:
            group = self.train_data["train"][item]
        except Exception:
            group = self.train_data[item]

        epoch = (
            int(self.trainer.state.epoch) if self.trainer.state.epoch is not None else 1
        )

        _hashed_seed = hash(item + self.trainer.args.seed)

        if "positive_passages" in group:
            query = group["query"]
            group_positives = group["positive_passages"]
            group_negatives = group["negative_passages"]

        else:  # this is for the minicpm data format
            query = group["query"][1]
            group_positives = [group["pos"][1]]  #!!!!!!!!!!!!!!!!!!!!
            group_negatives = group["neg"][1:]

        formated_query = format_query(query, self.data_args.query_prefix)
        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        formated_passages.append(
            format_passage(pos_psg, "", self.data_args.passage_prefix)
        )

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset : _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(
                format_passage(neg_psg, "", self.data_args.passage_prefix)
            )

        return formated_query, formated_passages


class TrainDatasetPreprocessed(Dataset):
    def __init__(self, data_args: DataArguments, trainer=None, is_eval=False):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )

        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def create_one_example(self, text_encoding: List[int], is_query=False):
        max_length = (
            self.data_args.query_max_len if is_query else self.data_args.passage_max_len
        )

        if self.data_args.append_eos_token:
            max_length -= 1

        item = self.tokenizer.prepare_for_model(
            text_encoding,
            truncation="only_first",
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group["query"]
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        group_positives = group["positives"]
        group_negatives = group["negatives"]

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        encoded_passages.append(self.create_one_example(pos_psg))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset : _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(self.create_one_example(neg_psg))

        return encoded_query, encoded_passages


class EncodeDataset(Dataset):

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.encode_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str]:
        text = self.encode_data[item]
        if self.data_args.encode_is_query:
            text_id = text["query_id"]
            formated_text = format_query(text["query"], self.data_args.query_prefix)
        else:
            text_id = text["docid"] if "docid" in text else text["text_id"]
            formated_text = format_passage(
                text["text"],
                text["title"],
                self.data_args.passage_prefix,
                self.data_args.add_markers,
            )

        return text_id, formated_text


class EncodeDataset_MARCOWeb(Dataset):

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args

        # simply record all possible lines and the corresponding cwid 
        self.dataset_dir = self.data_args.dataset_path

        self.encode_data = []
        if self.data_args.encode_is_query:
            # query: directly read csv 
            data = pd.read_csv(self.data_args.dataset_path, sep='\t', header=None)
            self.encode_data = [
                {'query_id': qid, "query": query} for qid, query in zip(list(data.iloc[:, 0]), list(data.iloc[:, 1]))
            ]
        else: 
            for lang in self.data_args.langs: 
                lang_dir = os.path.join(self.dataset_dir, lang)
                num_json_shards = len(os.listdir(lang_dir)) // 2
                for jsongz_id in range(0, num_json_shards):
                    jjsongz_id = str(jsongz_id).zfill(2)
                    jsongz_record_path = os.path.join(lang_dir, f"{lang}-{jjsongz_id}.offset")
                    with open(jsongz_record_path, 'r') as fp:
                        total_lines_in_jsongz = len(fp.readlines()) - 1 # extra lines per file 
                        # record all possible id in the json 
                        for doc_id in range(total_lines_in_jsongz): 
                            ddoc_id = str(doc_id).zfill(5)
                            self.encode_data.append(f"clueweb22-en-{jjsongz_id}-{ddoc_id}")
                            
        logger.info(f"EncodeDataset_MARCOWeb total length: {len(self.encode_data)}")

        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = create_shards(
                data=self.encode_data, 
                num_shards=self.data_args.dataset_number_of_shards, 
                index=self.data_args.dataset_shard_index
            )
        logger.info(f"EncodeDataset_MARCOWeb shard {self.data_args.dataset_shard_index} length: {len(self.encode_data)}")


    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str]:

        if self.data_args.encode_is_query:
            # query processing
            text = self.encode_data[item]
            text_id = text["query_id"]
            formated_text = format_query(text["query"], self.data_args.query_prefix)
        else:
            # document processing 
            cweb_doc_id = self.encode_data[item]
            clueweb_api = ClueWeb22Api(cweb_doc_id, self.dataset_dir)

            clean_txt = eval(clueweb_api.get_marcoweb_clean_text())
            text_id = clean_txt["ClueWeb22-ID"]
            content = clean_txt["Clean-Text"]
            title = content.split('\n')[0].replace("\n", "").replace("\t", "").replace("\r", "").replace("\'", "").replace("\"", "").strip()
            content = content.replace("\n", "").replace("\t", "").replace("\r", "").replace("\'", "").replace("\"", "").strip()
                
            formated_text = format_passage(
                content,
                title,
                self.data_args.passage_prefix,
                self.data_args.add_markers,
            )

        return text_id, formated_text