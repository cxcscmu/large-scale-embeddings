#!/bin/bash
#SBATCH --job-name=fw_0
#SBATCH --nodelist=boston-1-24
#SBATCH --partition=ssd
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=infinite
#SBATCH --output=node_log/fw_node_0.log
#SBATCH --error=node_log/fw_node_0.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting FineWeb Search Server on Shard 0 at 1-24..."

source ~/.bashrc
conda activate ethanenv
python fw_search_api/fw_node0.py