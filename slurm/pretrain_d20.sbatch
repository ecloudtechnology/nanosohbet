#!/bin/bash
#SBATCH --job-name=nanosohbet_d20
#SBATCH --partition=a100x4q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/ari/users/ytanis/.cache/nanochat
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
PY=/ari/users/ytanis/envs/nanochat/bin
cd /ari/users/ytanis/nanosohbet-repo

echo "Dugum: $(hostname)  Baslangic: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY/torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=20 \
    --device-batch-size=16 \
    --core-metric-every=-1 \
    --sample-every=2000 \
    --eval-every=250 \
    --save-every=2000 \
    --run=dummy

echo "Bitis: $(date)"
