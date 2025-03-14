# large-scale-embeddings


This repo leverages [Tevatron](https://github.com/texttron/tevatron) as the backbone code for dense retrieval training and inference.

Add-ons:

- [x] Distributed (single node) GPU search for fast exact search experiments.  
- [ ] VLLM inference.  

## Environment

`conda env create -f environment.yml`  
`conda activate minicpmembed`  
`cd tevatron`  
`pip install -e .`  

Create a local.env file where you set the following environment variable:

`HF_HOME=...`

This should be the path for the huggingface cache. set it accordingly to cluster specifications.  

In the future, if needed, HF_TOKEN should be set here too.


## Inference: MiniCPM-Embedding on ClueWeb documents

Check experiments folder for details on the scripts.

From root, run:

`sbatch experiments/encode_clueweb_docs_minicpm.sh`  
`sbatch experiments/encode_example_queries.sh`  
`sbatch experiments/search.sh`  

Some paths are currently hard-coded within the files.

### Dataset formats:

**Queries:** jsonl, `{"query_id": identifier, "query": text}`

**Documents:** jsonl, `{"docid": identifier, "text": text, "title": title}`
- If there is no title, set it to "", but code expects this key.

Also works with huggingface datasets out-of-the-box if in Tevatron format. For example:
- [Corpus](https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus)
- [Queries](https://huggingface.co/datasets/Tevatron/msmarco-passage)