#!/bin/bash
#SBATCH --job-name=mapping
#SBATCH --partition=cpu
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00
#SBATCH --output=node_log/mappinp_%j.log
#SBATCH --error=node_log/mappinp_%j.err


cd /bos/usr0/jening/PycharmProjects/DiskANN_Search

source ~/.bashrc
conda activate encode

python playground.py
