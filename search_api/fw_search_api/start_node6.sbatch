#!/bin/bash
#SBATCH --job-name=fw_6
#SBATCH --nodelist=boston-1-29
#SBATCH --partition=ssd
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=infinite
#SBATCH --output=node_log/fw_node_6.log
#SBATCH --error=node_log/fw_node_6.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting FineWeb Search Server on Shard 6 at 1-29..."

source ~/.bashrc
conda activate ethanenv
python fw_search_api/fw_node6.py