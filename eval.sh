#!/usr/bin/env bash
set -x

export NCCL_LL_THRESHOLD=0

CONFIG=$1
GPUS=$2
CPUS=$[GPUS*4]

CONFIG_NAME=${CONFIG##*/}
CONFIG_NAME=${CONFIG_NAME%.*}

OUTPUT_DIR="/data/VL-LTR/test/${CONFIG_NAME}"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir ${OUTPUT_DIR}
fi

python -u main.py \
    --num_workers 4 \
    --resume "${OUTPUT_DIR}/checkpoint.pth" \
    --output-dir $OUTPUT_DIR \
    --config $CONFIG ${@:4} \
    --eval \
    2>&1 | tee -a $OUTPUT_DIR/train.log