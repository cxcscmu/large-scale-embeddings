#!/bin/bash
#SBATCH --job-name=gt_3
#SBATCH --partition=cpu
#SBATCH --mem=84G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --output=node_log/gt_com_3.log
#SBATCH --error=node_log/gt_com_3.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"


echo "Job Starts"
cd /bos/usr0/jening/PycharmProjects/DiskANN_Search/

shard=3

# ######### Compute KNNS result for ANNS Benchmarking ########### # 
corpus_path="/bos/tmp2/jening/cw22_embeddings/clueweb_b_en/bin/cw22-b-en.$shard.bin"

query_path="/bos/usr0/jening/PycharmProjects/DiskANN_Search/data/researchy_questions/q_emb/queries_test_cleaned.bin"

gt_path="/bos/usr0/jening/PycharmProjects/DiskANN_Search/data/researchy_questions/q_emb/rq_test_cleaned_gt_$shard.bin"

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

./rest_api/apps/utils/compute_groundtruth  --data_type float --dist_fn mips \
        --base_file $corpus_path \
        --query_file $query_path \
        --gt_file $gt_path \
        --K $K


echo "Job Ends"