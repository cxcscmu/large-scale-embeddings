#!/bin/bash

#SBATCH --job-name=fineweb_6
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=128000M
#SBATCH --time=infinite
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --mail-type=END
#SBATCH --mail-user="your.email@your.domain"

R=80
L=120
B=64
M=64
T=16

# TODO: Change to your path
SHARD=6
EMB_PATH=/bos/tmp2/jening/fineweb_embeddings/bin
EMB_FILE=fineweb.${SHARD}.bin
INDEX_PATH=/bos/tmp2/jening/fineweb_embeddings/index
INDEX_PREFIX=fineweb-${SHARD}


echo "On node $(hostname)"
echo "Creating index for shard $SHARD"
echo "R=$R, L=$L, B=$B, M=$M, T=$T"
echo "EMB_PATH=$EMB_PATH"
echo "EMB_FILE=$EMB_FILE"
echo "INDEX_PATH=$INDEX_PATH"
echo "INDEX_PREFIX=$INDEX_PREFIX"

# rm -rf $INDEX_PATH
mkdir -p $INDEX_PATH

./build/apps/build_disk_index \
    --data_type float \
    --dist_fn mips \
    -R $R \
    -L $L \
    -B $B \
    -M $M \
    -T $T \
    --data_path $EMB_PATH/$EMB_FILE \
    --index_path_prefix $INDEX_PATH/$INDEX_PREFIX