#!/bin/bash

hostname

NODE_RANK=$1
MASTER_ADDR=$2
MASTER_PORT=6000
export TORCH_DISTRIBUTED_DEBUG=INFO

. conf.sh

echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "SAVE_DIR: $SAVE_DIR"

torchrun --nproc_per_node $NPROC_PER_NODE \
         --nnodes $NNODES \
         --node_rank $NODE_RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         main.py finetune --epochs 10 \
		                    --data_path exampledata/finetune/tox21.csv \
							--features_path exampledata/finetune/tox21.npz \
							--save_dir $SAVE_DIR \
							--checkpoint_path ./grover_large.pt \
							--split_type scaffold_balanced \
							--ensemble_size 1 \
							--num_folds 1 \
							--no_features_scaling \
                            --init_lr 0.00004 \
							--max_lr 0.0004 \
							--final_lr 0.00006 \
							--dropout 0.2 \
							--ffn_hidden_size 700 \
							--ffn_num_layers 2 \
							--attn_hidden 128 \
							--attn_out 8 \
							--dist_coff 0.05 \
							--bond_drop_rate 0 \
							--torchddp