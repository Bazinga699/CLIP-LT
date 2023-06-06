#!/usr/bin/env bash
set -x

export NCCL_LL_THRESHOLD=0
export MKL_SERVICE_FORCE_INTEL=1

CONFIG=$1
GPUS=$2
CPUS=$[GPUS*4]
NNODES=$3
RANK=$4
ADDR=$5
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi

CONFIG_NAME=${CONFIG##*/}
CONFIG_NAME=${CONFIG_NAME%.*}

OUTPUT_DIR="/data//data/VL_CLS/VL_LTR/${CONFIG_NAME}"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p ${OUTPUT_DIR}
fi

python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$RANK --master_addr=$ADDR --nproc_per_node=$GPUS main.py \
    --num_workers 4 \
    --config $CONFIG ${@:6} \
    2>&1 | tee -a ${OUTPUT_DIR}/train.log
