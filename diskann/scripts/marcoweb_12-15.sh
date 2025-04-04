#!/bin/bash

#SBATCH --job-name=ANN_Server_12-15
#SBATCH --partition=cpu
#SBATCH --nodelist=boston-1-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=128000M
#SBATCH --time=infinite
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 

echo "Starting job on $(date)"
echo "Running on node $(hostname)"
echo "Running in directory $(pwd)"
echo "Using SLURM job ID $SLURM_JOB_ID"

shard=12-15

if [ $shard = "0-3" ]; then
    prefix=marcoweb_minicpm-light_index_0-3_R80_L120_B64_M64_T8
    nodelist=boston-1-9
elif [ $shard = "4-7" ]; then
    prefix=marcoweb_minicpm-light_index_4-7_R80_L120_B64_M80_T16
    nodelist=boston-1-24
elif [ $shard = "8-11" ]; then
    prefix=marcoweb_minicpm-light_index_8-11_R80_L120_B64_M80_T16
    nodelist=boston-1-10
elif [ $shard = "12-15" ]; then
    prefix=marcoweb_minicpm-light_index_12-15_R80_L120_B64_M80_T16
    nodelist=boston-1-23
else
    echo "Invalid argument. Please use 0-3, 4-7, 8-11, or 12-15."
    exit 1
fi


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/karrym/software/casablanca-tevinw/build.debug/Release/Binaries

./build-tevinw/apps/restapi/ssd_server \
    --address http://0.0.0.0:8008 \
    --data_type float \
    --index_path_prefix /ssd1/karrym/DiskANN/data/clueweb/$prefix \
    --num_nodes_to_cache 10000 \
    --num_threads 16