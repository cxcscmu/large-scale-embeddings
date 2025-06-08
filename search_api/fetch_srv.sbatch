#!/bin/bash
#SBATCH --job-name=fetch_srv
#SBATCH --nodelist=boston-1-10
#SBATCH --partition=cpu
#SBATCH --mem=24000M
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --time=infinite
#SBATCH --output=node_log/fetch_srv.log
#SBATCH --error=node_log/fetch_srv.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

cd /bos/usr0/jening/PycharmProjects/DiskANN_Search

source ~/.bashrc
conda activate encode

python fetch_srv.py
