#!/bin/bash

# Generic SLURM script for CW22 search nodes
# Parameters are passed via environment variables from the main script

echo "Starting CW22 Search Server on Node $NODE_ID..."
echo "Index Directory: $INDEX_DIR"
echo "Port: $PORT"

# Activate environment
source ~/.bashrc
conda activate ethanenv

# Run the generic Python script with parameters
python cw22_search_api/cw22_node_generic.py --node-id "$NODE_ID" --index-dir "$INDEX_DIR" --port "$PORT" --index-prefix "$INDEX_PREFIX"