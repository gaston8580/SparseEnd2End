# !/usr/bin/env bash
CONFIG=$1
GPUS="$MA_NUM_GPUS"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
PORT="6060"
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
JOB_ID="1234"

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
torchrun \
--nnodes=$NNODES \
--node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR \
--nproc_per_node=$GPUS \
--master_port=$PORT \
$(dirname "$0")/script/train.py $CONFIG --no-validate --launcher pytorch ${@:3}

