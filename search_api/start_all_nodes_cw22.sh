#!/bin/bash

echo "Submitting Node 0 task..."
sbatch cw22_search_api/start_node0.sbatch
sleep 1

echo "Submitting Node 1 task..."
sbatch cw22_search_api/start_node1.sbatch
sleep 1

echo "Submitting Node 2 task..."
sbatch cw22_search_api/start_node2.sbatch
sleep 1

echo "Submitting Node 3 task..."
sbatch cw22_search_api/start_node3.sbatch
sleep 1

echo "All nodes have been submitted!"