#!/bin/bash
#SBATCH --job-name=fw_4
#SBATCH --nodelist=boston-1-23
#SBATCH --partition=ssd
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=infinite
#SBATCH --output=node_log/fw_node_4.log
#SBATCH --error=node_log/fw_node_4.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting FineWeb Search Server on Shard 4 at 1-23..."

source ~/.bashrc
conda activate ethanenv
python fw_search_api/fw_node4.py