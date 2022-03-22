#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

set -x

PARTITION=Zoetrope
JOB_NAME=Pare
CONFIG=/mnt/cache/wangyanjun/mmhuman3d/configs/pare/hrnet_w32_conv_pare_mixed.py
WORK_DIR=/mnt/lustre/wangyanjun/pare_log
GPUS=2
GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=--no-validate

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
