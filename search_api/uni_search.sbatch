#!/bin/bash
#SBATCH --job-name=uni_search
#SBATCH --nodelist=boston-2-33
#SBATCH --partition=gpu
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=infinite
#SBATCH --output=node_log/searcher_%j.log
#SBATCH --error=node_log/searcher_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting Uni Search Service at $(date)"
cd /bos/usr0/jening/PycharmProjects/DiskANN_Search

source ~/.bashrc
conda activate encode

# 1. Start the SEARCH_SRV Flask service in the background, logging to searcher.log
nohup python3 uni_search_srv.py > node_log/searcher.log 2>&1 &
SEARCH_SRV_PID=$!

# 2. Wait for the SEARCH_SRV service to fully start
sleep 5

# Start cloudflared tunnel
nohup ~/bin/cloudflared tunnel run clueweb22-tunnel > node_log/tunnel.log 2>&1 &


wait $SEARCH_SRV_PID
