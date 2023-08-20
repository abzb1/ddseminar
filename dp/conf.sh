#!/bin/bash         

# configuration: container 
CONTAINER_IMAGE_PATH="$HOME/ddseminar/image/megatron-latest.sqsh"
CONTAINER_PATH="/scratch/enroot/$UID/data/megatron-latest"
CONTAINER_NAME="megatron-latest"

# configuration: distributed learning
NPROC_PER_NODE=4
NNODES=2
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
BATCH_SIZE=32
MASTER_PORT=6000
