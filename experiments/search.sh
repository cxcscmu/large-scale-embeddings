#!/bin/bash
#SBATCH --job-name=search
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate minicpmembed

EMBEDDING_OUTPUT_DIR=/data/group_data/cx_group/large_scale_index/temp/MiniCPM-Embedding-Light

set -f && python -m tevatron.retriever.driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/sample_queries.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/corpus.cweb.*.sample.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/sample.run.txt
