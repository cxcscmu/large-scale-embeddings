#!/bin/bash
#SBATCH --job-name=qwen3_search
#SBATCH --nodelist=boston-2-33
#SBATCH --partition=gpu
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=infinite
#SBATCH --output=node_log/qwen3_%j.log
#SBATCH --error=node_log/qwen3_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting CW22 Search Service at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"

cd /bos/usr0/jening/PycharmProjects/DiskANN_Search

source ~/.bashrc
conda activate encode

# Start the SEARCH_SRV Flask service in the background, logging to searcher.log
nohup python3 cw22_qwen_search_srv.py > ./node_log/qwen3_search_srv_${SLURM_JOB_ID}.log 2>&1 &
SEARCH_SRV_PID=$!

wait $SEARCH_SRV_PID