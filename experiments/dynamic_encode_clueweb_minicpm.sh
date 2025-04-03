#!/bin/bash
#SBATCH --job-name=clueweb22_minicpm
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=preempt  
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3


eval "$(conda shell.bash hook)"
conda activate minicpmembed
module load cuda-12.1

set -o allexport
source local.env
set +o allexport

echo $HF_HOME


PATH_TO_MODEL=openbmb/MiniCPM-Embedding-Light

EMBEDDING_OUTPUT_DIR=/data/group_data/cx_group/ann_index/embeds/clueweb22b/MiniCPM-Embedding-Light

mkdir -p $EMBEDDING_OUTPUT_DIR

shard=${SLURM_ARRAY_TASK_ID}
echo $shard
num_total_shard=16

local_bz=2048


# corpus encoding
PATH_TO_CORPUS="/data/datasets/clueweb22/ClueWeb22_B" 
# TODO: langs default to all, otherwise "en" "de" .etc
python -m tevatron.retriever.driver.encode \
    --clueweb_api_dataset True \
    --langs "en" \
    --output_dir $EMBEDDING_OUTPUT_DIR \
    --bf16 \
    --model_name_or_path $PATH_TO_MODEL \
    --dataset_cache_dir $HF_HOME \
    --cache_dir $HF_HOME \
    --query_prefix "Instruction: Given a web search query, retrieve relevant passages that answer the query. Query: " \
    --passage_prefix "" \
    --pooling avg \
    --normalize \
    --per_device_eval_batch_size $local_bz \
    --query_max_len 32 \
    --passage_max_len 512 \
    --dataset_path $PATH_TO_CORPUS \
    --add_markers False \
    --dataset_number_of_shards $num_total_shard \
    --dataset_shard_index ${shard} \
    --inference_save_step 20 \
    --encode_output_path $EMBEDDING_OUTPUT_DIR/clueweb22-corpus.cweb.${shard}.pkl
