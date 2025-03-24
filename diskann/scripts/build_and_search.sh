#!/bin/bash
#SBATCH --job-name=diskann_usage 
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=general 
#SBATCH --nodes=1

#SBATCH --mem=200G
#SBATCH --gres=gpu:0

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

#SBATCH --time=96:00:00


echo "Job Starts"


# TODO
index_root_dir=""
dataset=""

# index parameters 
R=100
LBuild=100
B=64
M=128
index_name="${dataset}_R${R}_L${LBuild}_B${B}_M${M}"
index_prefix="index_" 


# ########### Build ############### #
# # build params: 
    # --data_path: the first 4 byte is the number of document vectors, 
        # the second 4 byte is the dimension of the dcoument vectors, 
        # following by the document embedding array written in bytes 
    # -index_path_prefix: you need to give a valid folder followed by a file prefix (index_)
    # -R: max node degree
    # -L: build list length
    # -M: build DRAM budget
    # -B: search DRAM budget

# corpus embeddings
corpus_path="" # TODO
index_dir="${index_root_dir}/${index_name}"
./DiskANN/build/apps/build_disk_index --data_type float --dist_fn mips \
        --data_path $corpus_path \
        --index_path_prefix ${index_dir}/${index_prefix} \
        -R $R -L $LBuild -B $B -M $M 


# ########### Search ############### #

# # search params: 
    # --query_file: the first 4 byte is the number of query vectors, 
        # the second 4 byte is the dimension of the query vectors, 
        # following by the query array written in bytes 
    # --gt_file (optional): the first 4 byte is the number of query vectors, 
        # the second 4 byte is the number of truth candidates 
        # following by the ground truth array of truth candidate indices in the corpus (integer) in bytes
    # -result_path: you need to give a valid folder followed by a file prefix (res_)
    # -K: retrieve the closest K ids for each query vector
    # -L: search list length 

query_path="" # TODO
gt_path=""

# search param 
K=100
LSearch=100

result_dir="${index_dir}/results/K${K}_L${LSearch}"
mkdir -p $result_dir

./DiskANN/build/apps/search_disk_index --data_type float --dist_fn mips \
    --index_path_prefix "${index_dir}/${index_prefix}" \
    --query_file $query_path \
    --gt_file $gt_path \
    -K $K -L $LSearch  \
    --result_path "${result_dir}/res_" \
    --num_nodes_to_cache 10000



echo "Job Ends"