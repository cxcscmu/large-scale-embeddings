#!/bin/bash
#SBATCH --job-name=cw22_searcher
#SBATCH --nodelist=boston-2-33
#SBATCH --partition=gpu
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=node_log/searcher_job_%j.log
#SBATCH --error=node_log/searcher_job_%j.err

echo "Starting CW22 Search Service at $(date)"
cd /bos/usr0/jening/PycharmProjects/DiskANN_Search

source ~/.bashrc
conda activate encode

# 1. Start the SEARCH_SRV Flask service in the background, logging to searcher.log
nohup python3 search_service.py > searcher.log 2>&1 &
SEARCH_SRV_PID=$!

# 2. Wait for the SEARCH_SRV service to fully start
sleep 5

# 3. Start the SSH reverse tunnel via serveo.net in the background
nohup ssh -R 80:localhost:51000 serveo.net > ssh_tunnel.log 2>&1 &

# 4. Wait for the Serveo connection to be established
sleep 5

# 5. Extract the public Serveo URL and write it to a file
SERVEO_URL=$(grep -o 'https://[a-zA-Z0-9]*\.serveo\.net' ssh_tunnel.log | head -n 1)

echo "$SERVEO_URL" > serveo_url.txt

if [[ -z "$SERVEO_URL" ]]; then
    echo "? Failed to retrieve Serveo URL. Check ssh_tunnel.log for details."
else
    echo "? Serveo is running at: $SERVEO_URL"
    echo "?? You can access the service at: ${SERVEO_URL}/search?query=CMU&k=3"
fi

echo "? CW22 Searcher Slurm job launched successfully at $(date)"


wait $SEARCH_SRV_PID
