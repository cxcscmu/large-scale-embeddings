#!/bin/bash
#SBATCH --job-name=fw_3
#SBATCH --nodelist=boston-1-28
#SBATCH --partition=ssd
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=infinite
#SBATCH --output=node_log/fw_node_3.log
#SBATCH --error=node_log/fw_node_3.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting FineWeb Search Server on Shard 3 at 1-28..."

source ~/.bashrc
conda activate ethanenv
python fw_search_api/fw_node3.py