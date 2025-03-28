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

On the root folder, create a local.env file where you set the following environment variable:

`HF_HOME=...`

This should be the path for the huggingface cache. set it accordingly to cluster specifications.  

In the future, if needed, HF_TOKEN should be set here too.


## Inference: MiniCPM-Embedding-Light on ClueWeb documents

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



## Memory-SSD Hybrid Retrieval: DiskANN 

DiskANN should be installed according to the instructions in the [official DiskANN opensource repository](https://github.com/microsoft/DiskANN/tree/main). 


### Data Preparation Helpers

The inputs and outputs of DiskANN follows the [DiskANN documentation](https://github.com/microsoft/DiskANN/blob/main/workflows/SSD_index.md). 

#### Document embeddings 

`diskann/utils.py: convert_encoded_pkl_to_binary()`: convert the pickle-packed file (an output file from the above inference section) containing a (embeds. docids) tuple into a binary embedding file following DiskANN format and a separated pickle file for the document ids. 

`diskann/utils.py: convert_encoded_pkls_to_binary()`: convert a list of pickle-packed files (output shards from the above inference section) each containing a (embeds. docids) tuple into a single binary embedding file following DiskANN format and a separated pickle file for the document ids. 

`diskann/utils.py: read_fbin()`: read a binary embedding file in DiskANN format. 

`diskann/utils.py: write_embed_to_binary()`: write an embedding array to a binary embedding file in DiskANN format. 


#### Groud Truth

`diskann/utils.py: retrieval_result_read()`: read the grouth truth file in DiskANN format, the e2e option specify whether a distance array will be read (ANNS scenario) or not (end-to-end qrel scenario). 

##### ANNS Truth

Modify the parameters with `#TODO` tags in `diskann/scripts/utils.sh` and run: 
`sbatch diskann/scripts/utils.sh`

##### End-to-End Truth

`diskann/utils.py: read_trec_qrels()`: read a trec_eval formatted qrel file and return a dictionary of qids and their corresponding qrels. 

`diskann/utils.py: write_qrels_to_binary()`: write a dictionary of qids and their corresponding qrels into a binary ground truth file supported by DiskANN.






### Index Build and Search
Modify the parameters with `#TODO` tags in `diskann/scripts/build_and_search.sh` and run: 
`sbatch diskann/scripts/build_and_search.sh`