#!/bin/bash
#SBATCH --job-name=fw_1
#SBATCH --nodelist=boston-1-25
#SBATCH --partition=ssd
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=infinite
#SBATCH --output=node_log/fw_node_1.log
#SBATCH --error=node_log/fw_node_1.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting FineWeb Search Server on Shard 1 at 1-25..."

source ~/.bashrc
conda activate ethanenv
python fw_search_api/fw_node1.py