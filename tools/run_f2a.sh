srun -p Sensetime \
    --job-name="python" \
    --gres=gpu:1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
    python -u run_f2a.py --exp_dir=$1