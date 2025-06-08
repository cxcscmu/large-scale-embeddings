#!/bin/bash
#SBATCH --job-name=trec_e
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=999:00:00
#SBATCH --output=node_log/trec_e_%j.log
#SBATCH --error=node_log/trec_e_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting CW22 trec eval at $(date)"
cd /bos/usr0/jening/PycharmProjects/DiskANN_Search

source ~/.bashrc
conda activate encode

python3 cw22_searcher.py

