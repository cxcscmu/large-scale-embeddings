# Adapted from Tevatron (https://github.com/texttron/tevatron)

from argparse import ArgumentParser
from transformers import AutoTokenizer, PreTrainedTokenizer
import os
import warnings
import random
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
import datasets
from dataclasses import dataclass
from typing import Dict, List, Any
import csv
import json

def find_all_markers(template: str):
    """
    Find all markers' names (quoted in "<>") in a template.
    """
    markers = []
    start = 0
    while True:
        start = template.find("<", start)
        if start == -1:
            break
        end = template.find(">", start)
        if end == -1:
            break
        markers.append(template[start + 1:end])
        start = end + 1
    return markers

def fill_template(template: str, data: Dict, markers: List[str] = None, allow_not_found: bool = False):
    """
    Fill a template with data.
    """
    if markers is None:
        markers = find_all_markers(template)
    for marker in markers:
        marker_hierarchy = marker.split(".")
        found = True
        content = data
        for marker_level in marker_hierarchy:
            content = content.get(marker_level, None)
            if content is None:
                found = False
                break
        if not found:
            if allow_not_found:
                warnings.warn("Marker '{}' not found in data. Replacing it with an empty string.".format(marker), RuntimeWarning)
                content = ""
            else:
                raise ValueError("Cannot find the marker '{}' in the data".format(marker))
        template = template.replace("<{}>".format(marker), str(content))
    return template 


@dataclass
class SimpleTrainPreProcessor:
    query_file: str
    collection_file: str
    tokenizer: PreTrainedTokenizer
    columns: str

    doc_max_len: int = 128
    query_max_len: int = 32
    title_field = 'title'
    text_field = 'text'
    query_field = 'text'
    doc_template: str = None
    query_template: str = None
    allow_not_found: bool = False

    def __post_init__(self):
        self.queries = self.read_queries(self.query_file)
        self.collection = datasets.load_dataset(
            'csv',
            data_files=self.collection_file,
            column_names=self.columns,
            delimiter='\t',
            #cache_dir="/data/datasets/hf_cache",
        )['train']

    @staticmethod
    def read_queries(queries):
        qmap = {}
        with open(queries) as f:
            for l in f:
                qid, qry = l.strip().split('\t')
                qmap[qid] = qry
        return qmap

    @staticmethod
    def read_qrel(relevance_file):
        qrel = {}
        with open(relevance_file, encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, _, docid, rel] in tsvreader:
                assert rel == "1"
                if topicid in qrel:
                    qrel[topicid].append(docid)
                else:
                    qrel[topicid] = [docid]
        return qrel

    def get_query(self, q):
        if self.query_template is None:
            query = self.queries[q]
        else:
            query = fill_template(self.query_template, data={self.query_field: self.queries[q]}, allow_not_found=self.allow_not_found)
        query_encoded = self.tokenizer.encode(
            query,
            add_special_tokens=False,
            max_length=self.query_max_len,
            truncation=True
        )
        return query_encoded

    def get_passage(self, p, split_token):
        entry = self.collection[int(p)] if p != "None" else self.collection[0]
        if "title" in self.columns:
            title = entry[self.title_field]
            title = "" if title is None else title
        
        else:
            title = ""
        body = entry[self.text_field]

        if not split_token:

            if self.doc_template is None:
                content = title + self.tokenizer.sep_token + body
            else:
                content = fill_template(self.doc_template, data=entry, allow_not_found=self.allow_not_found)

            passage_encoded = self.tokenizer.encode(
                content,
                add_special_tokens=False,
                max_length=self.doc_max_len,
                truncation=True
            )

            return passage_encoded

        else: 
            split_body = body.split(split_token)
            contents = []
            for text in split_body:
                if self.doc_template is None:
                    content = title + self.tokenizer.sep_token + text
                else:
                    content = fill_template(self.doc_template, data={self.title_field: title, self.text_field: text}, allow_not_found=self.allow_not_found)

                contents.append(content)
            
            encoded = []
            for content in contents:
                passage_encoded = self.tokenizer.encode(
                    content,
                    add_special_tokens=False,
                    max_length=self.doc_max_len,
                    truncation=True
                )
                encoded.append(passage_encoded)
            
            return encoded


        
    def process_one(self, train):
        q, pp, nn, split_token = train
        train_example = {
            'query': self.get_query(q),
            'positives': [self.get_passage(p, split_token) for p in pp],
            'negatives': [self.get_passage(n, split_token) for n in nn],
        }

        return json.dumps(train_example)


random.seed(17121998)
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--negative_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--doc_template', type=str, default=None)
parser.add_argument('--query_template', type=str, default=None)
parser.add_argument('--columns', type=str, default="text_id,title,text")

parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--truncate_q', type=int, default=32)
parser.add_argument('--n_sample', type=int, default=30)
parser.add_argument('--mp_chunk_size', type=int, default=500)
parser.add_argument('--shard_size', type=int, default=45000)
parser.add_argument('--split_sentences', type=str, default=None)

args = parser.parse_args()


qrel = SimpleTrainPreProcessor.read_qrel(args.qrels)


def read_line(l):
    q, nn = l.strip().split('\t')
    nn = nn.split(',')
    random.shuffle(nn)
    return q, qrel[q], nn[:args.n_sample], args.split_sentences


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = SimpleTrainPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
    tokenizer=tokenizer,
    doc_max_len=args.truncate,
    query_max_len=args.truncate_q,
    doc_template=args.doc_template,
    query_template=args.query_template,
    allow_not_found=True,
    columns=args.columns.split(",")
)

counter = 0
shard_id = 0
f = None
os.makedirs(args.save_to, exist_ok=True)

with open(args.negative_file) as nf:
    pbar = tqdm(map(read_line, nf))
    with Pool() as p:
        for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
            counter += 1
            if f is None:
                f = open(os.path.join(args.save_to, f'split{shard_id:02d}.jsonl'), 'w')
                pbar.set_description(f'split - {shard_id:02d}')
            f.write(x + '\n')

            if counter == args.shard_size:
                f.close()
                f = None
                shard_id += 1
                counter = 0

if f is not None:
    f.close()