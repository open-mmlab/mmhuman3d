CONFIG=$1
WORK_DIR=$2
CHECKPOINT=$3
GPUS=$4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG --work-dir=${WORK_DIR} $CHECKPOINT --launcher pytorch ${@:5}
