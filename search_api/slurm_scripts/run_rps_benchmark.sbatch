#!/bin/bash
#SBATCH --job-name=cw22_rps
#SBATCH --partition=cpu
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=999:00:00
#SBATCH --output=node_log/rps_%j.log
#SBATCH --error=node_log/rps_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting CW22 time benchmark at $(date)"
cd /bos/usr0/jening/PycharmProjects/DiskANN_Search

source ~/.bashrc
conda activate encode

python3 ./utils/benchmark_search_rps.py

