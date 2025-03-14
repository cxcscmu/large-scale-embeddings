#!/bin/bash
#SBATCH --job-name=inferece_q
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00

export HF_HOME=/data/group_data/cx_group/query_generation_data/hf_cache/
eval "$(conda shell.bash hook)"
conda activate minicpmembed
module load cuda-12.1

set -o allexport
source local.env
set +o allexport


echo $HF_HOME

PATH_TO_QUERIES="/data/group_data/cx_group/large_scale_index/temp/MiniCPM-Embedding/sample_queries.jsonl"

PATH_TO_MODEL=openbmb/MiniCPM-Embedding

EMBEDDING_OUTPUT_DIR=/data/group_data/cx_group/large_scale_index/temp/MiniCPM-Embedding

mkdir -p $EMBEDDING_OUTPUT_DIR

python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path $PATH_TO_MODEL \
    --bf16 \
    --pooling avg \
    --dataset_cache_dir $HF_HOME \
    --cache_dir $HF_HOME \
    --normalize \
    --query_prefix "Instruction: Given a web search query, retrieve relevant passages that answer the query. Query: " \
    --passage_prefix "" \
    --encode_is_query \
    --per_device_eval_batch_size 300 \
    --query_max_len 64 \
    --passage_max_len 512 \
    --dataset_path $PATH_TO_QUERIES \
    --encode_output_path $EMBEDDING_OUTPUT_DIR/sample_queries.pkl





