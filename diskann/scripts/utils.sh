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


# ######### Compute KNNS result for ANNS Benchmarking ########### # 

query_path="" # TODO
gt_path="" # TODO
K=100
# # compute ground truth params: 
    # --query_file: the first 4 byte is the number of query vectors, 
        # the second 4 byte is the dimension of the query vectors, 
        # following by the query array written in bytes 
    # --gt_file: the first 4 byte is the number of query vectors, 
        # the second 4 byte is the number of truth candidates 
        # following by the ground truth array of truth candidate indices in the corpus (integer) in bytes
        # following by the distances array of truth candidates to the query (float) in bytes 
    # -K: retrieve the closest K ids for each query vector

./DiskANN/build/apps/utils/compute_groundtruth  --data_type float --dist_fn mips \
        --base_file $corpus_path \
        --query_file $query_path \
        --gt_file $gt_path \
        --K $K


echo "Job Ends"