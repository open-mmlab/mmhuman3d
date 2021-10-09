#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
INPUT=$3
INPUT_TYPE=$4
CONFIG=$5
BODY_MODEL_DIR=$6
OUTPUT=$7
SHOW_PATH=$8
GPUS=1 # Only support single GPU currently
GPUS_PER_NODE=1
CPUS_PER_TASK=1
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:9}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/smplify.py \
    --use_one_betas_per_video \
    --input=${INPUT} \
    --input_type=${INPUT_TYPE} \
    --config=${CONFIG} \
    --body_model_dir=${BODY_MODEL_DIR} \
    --output=${OUTPUT} \
    --show_path=${SHOW_PATH} ${PY_ARGS}
