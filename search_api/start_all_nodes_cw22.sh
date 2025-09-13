#!/bin/bash

# Configuration for all nodes
# Each line: node_id,job_name,nodelist,index_dir,port,index_prefix
NODES_CONFIG=(
    # en00
    "0,cw22_0_0,boston-1-24,/ssd2/cw22_ann_index/cw22_b_en/R80L120-0,51100,cw22_b_en"
    "1,cw22_0_1,boston-1-25,/ssd2/cw22_ann_index/cw22_b_en/R80L120-1,51100,cw22_b_en"
    "2,cw22_0_2,boston-1-26,/ssd1/cw22_ann_index/cw22_b_en/R80L120-2,51100,cw22_b_en"
    "3,cw22_0_3,boston-1-28,/ssd2/cw22_ann_index/cw22_b_en/R80L120-3,51100,cw22_b_en"
    # en01
    "4,cw22_1_0,boston-1-25,/ssd2/cw22a_ann_index/cw22_a_en01/R80L120-0,51101,cw22_0"
    "5,cw22_1_1,boston-1-26,/ssd2/cw22a_ann_index/cw22_a_en01/R80L120-1,51101,cw22_1"
    "6,cw22_1_2,boston-1-27,/ssd2/cw22a_ann_index/cw22_a_en01/R80L120-2,51101,cw22_2"
    "7,cw22_1_3,boston-1-28,/ssd2/cw22a_ann_index/cw22_a_en01/R80L120-3,51101,cw22_3"
    # en02
    "8,cw22_2_0,boston-1-23,/ssd2/cw22a_ann_index/cw22_a_en02/R80L120-0,51102,cw22_0"
    "9,cw22_2_1,boston-1-24,/ssd2/cw22a_ann_index/cw22_a_en02/R80L120-1,51102,cw22_1"
    "10,cw22_2_2,boston-1-27,/ssd2/cw22a_ann_index/cw22_a_en02/R80L120-2,51102,cw22_2"
    "11,cw22_2_3,boston-1-25,/ssd2/cw22a_ann_index/cw22_a_en02/R80L120-3,51102,cw22_3"
    # en03
    "12,cw22_3_0,boston-1-23,/ssd2/cw22a_ann_index/cw22_a_en03/R80L120-0,51103,cw22_0"
    "13,cw22_3_1,boston-1-24,/ssd2/cw22a_ann_index/cw22_a_en03/R80L120-1,51103,cw22_1"
    "14,cw22_3_2,boston-1-27,/ssd2/cw22a_ann_index/cw22_a_en03/R80L120-2,51103,cw22_2"
    "15,cw22_3_3,boston-1-26,/ssd2/cw22a_ann_index/cw22_a_en03/R80L120-3,51103,cw22_3"
    # en04
    "16,cw22_4_0,boston-1-23,/ssd2/cw22a_ann_index/cw22_a_en04/R80L120-0,51104,cw22_0"
    "17,cw22_4_1,boston-1-24,/ssd2/cw22a_ann_index/cw22_a_en04/R80L120-1,51104,cw22_1"
    "18,cw22_4_2,boston-1-27,/ssd2/cw22a_ann_index/cw22_a_en04/R80L120-2,51104,cw22_2"
    "19,cw22_4_3,boston-1-28,/ssd2/cw22a_ann_index/cw22_a_en04/R80L120-3,51104,cw22_3"
    # en05
    "20,cw22_5_0,boston-1-23,/ssd2/cw22a_ann_index/cw22_a_en05/R80L120-0,51105,cw22_0"
    "21,cw22_5_1,boston-1-24,/ssd2/cw22a_ann_index/cw22_a_en05/R80L120-1,51105,cw22_1"
    "22,cw22_5_2,boston-1-27,/ssd2/cw22a_ann_index/cw22_a_en05/R80L120-2,51105,cw22_2"
    "23,cw22_5_3,boston-1-29,/ssd2/cw22a_ann_index/cw22_a_en05/R80L120-3,51105,cw22_3"
    # en06
    "24,cw22_6_0,boston-1-23,/ssd2/cw22a_ann_index/cw22_a_en06/R80L120-0,51106,cw22_0"
    "25,cw22_6_1,boston-1-24,/ssd2/cw22a_ann_index/cw22_a_en06/R80L120-1,51106,cw22_1"
    "26,cw22_6_2,boston-1-27,/ssd2/cw22a_ann_index/cw22_a_en06/R80L120-2,51106,cw22_2"
    "27,cw22_6_3,boston-1-29,/ssd2/cw22a_ann_index/cw22_a_en06/R80L120-3,51106,cw22_3"
    # en07
    "28,cw22_7_0,boston-1-25,/ssd2/cw22a_ann_index/cw22_a_en07/R80L120-0,51107,cw22_0"
    "29,cw22_7_1,boston-1-26,/ssd2/cw22a_ann_index/cw22_a_en07/R80L120-1,51107,cw22_1"
    "30,cw22_7_2,boston-1-28,/ssd2/cw22a_ann_index/cw22_a_en07/R80L120-2,51107,cw22_2"
    "31,cw22_7_3,boston-1-29,/ssd2/cw22a_ann_index/cw22_a_en07/R80L120-3,51107,cw22_3"
    # en08
    "32,cw22_8_0,boston-1-25,/ssd2/cw22a_ann_index/cw22_a_en08/R80L120-0,51108,cw22_0"
    "33,cw22_8_1,boston-1-26,/ssd2/cw22a_ann_index/cw22_a_en08/R80L120-1,51108,cw22_1"
    "34,cw22_8_2,boston-1-28,/ssd2/cw22a_ann_index/cw22_a_en08/R80L120-2,51108,cw22_2"
    "35,cw22_8_3,boston-1-29,/ssd2/cw22a_ann_index/cw22_a_en08/R80L120-3,51108,cw22_3"
    # en09
    "36,cw22_9_0,boston-1-25,/ssd2/cw22a_ann_index/cw22_a_en09/R80L120-0,51109,cw22_0"
    "37,cw22_9_1,boston-1-26,/ssd2/cw22a_ann_index/cw22_a_en09/R80L120-1,51109,cw22_1"
    "38,cw22_9_2,boston-1-28,/ssd2/cw22a_ann_index/cw22_a_en09/R80L120-2,51109,cw22_2"
    "39,cw22_9_3,boston-1-29,/ssd2/cw22a_ann_index/cw22_a_en09/R80L120-3,51109,cw22_3"
)

# Common SLURM parameters
PARTITION="ssd"
MEM="16G"
NTASKS_PER_NODE="1"
CPUS_PER_TASK="2"
TIME="infinite"
MAIL_USER="ethanning@cmu.edu"

echo "Starting CW22 Search Service on multiple nodes..."

# Loop through each node configuration
for config in "${NODES_CONFIG[@]}"; do
    # Parse configuration
    IFS=',' read -r node_id job_name nodelist index_dir port index_prefix <<< "$config"
    
    echo "Submitting Node $node_id task on $nodelist..."
    
    # Submit job with parameters
    sbatch \
        --job-name="$job_name" \
        --nodelist="$nodelist" \
        --partition="$PARTITION" \
        --mem="$MEM" \
        --ntasks-per-node="$NTASKS_PER_NODE" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --time="$TIME" \
        --output="node_log/cw22/${job_name}.log" \
        --error="node_log/cw22/${job_name}.err" \
        --mail-type=END \
        --mail-user="$MAIL_USER" \
        --export=NODE_ID="$node_id",INDEX_DIR="$index_dir",PORT="$port",PORT="$port",INDEX_PREFIX="$index_prefix" \
        cw22_search_api/start_node_generic.sh
    
    # Small delay between submissions
    sleep 1
done

echo "All nodes have been submitted!"
echo "Check logs in node_log/ directory for status"