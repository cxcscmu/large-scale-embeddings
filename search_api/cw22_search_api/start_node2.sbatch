#!/bin/bash
#SBATCH --job-name=cw22_n2
#SBATCH --nodelist=boston-1-26
#SBATCH --partition=ssd
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=infinite
#SBATCH --output=node_log/search_node_2.log
#SBATCH --error=node_log/search_node_2.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting CW22 Search Server on Shard 2 at 1-26..."

source ~/.bashrc
conda activate ethanenv
python cw22_search_api/cw22_node2.py