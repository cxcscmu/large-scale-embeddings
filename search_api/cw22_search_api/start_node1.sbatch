#!/bin/bash
#SBATCH --job-name=cw22_n1
#SBATCH --nodelist=boston-1-25
#SBATCH --partition=ssd
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=infinite
#SBATCH --output=node_log/search_node_1.log
#SBATCH --error=node_log/search_node_1.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting CW22 Search Server on Shard 1 at 1-25..."

source ~/.bashrc
conda activate ethanenv
python cw22_search_api/cw22_node1.py