#!/bin/bash

hostname

NODE_RANK=$1
MASTER_ADDR=$2
MASTER_PORT=7000

. conf.sh

echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "BATCH_SIZE: $BATCH_SIZE"

torchrun --nproc_per_node $NPROC_PER_NODE \
         --nnodes $NNODES \
         --node_rank $NODE_RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         train.py -e 5 \
                  -b $BATCH_SIZE \
                  -np $NPROC_PER_NODE \
                  -n $NNODES
