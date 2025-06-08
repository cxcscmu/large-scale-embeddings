#!/bin/bash

echo "Submitting Node 0 task..."
sbatch fw_search_api/start_node0.sbatch
sleep 1

echo "Submitting Node 1 task..."
sbatch fw_search_api/start_node1.sbatch
sleep 1

echo "Submitting Node 2 task..."
sbatch fw_search_api/start_node2.sbatch
sleep 1

echo "Submitting Node 3 task..."
sbatch fw_search_api/start_node3.sbatch
sleep 1

echo "Submitting Node 4 task..."
sbatch fw_search_api/start_node4.sbatch
sleep 1

echo "Submitting Node 5 task..."
sbatch fw_search_api/start_node5.sbatch
sleep 1

echo "Submitting Node 6 task..."
sbatch fw_search_api/start_node6.sbatch
sleep 1

echo "All nodes have been submitted!"