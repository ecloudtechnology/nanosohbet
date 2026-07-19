#!/bin/bash
#SBATCH --job-name=nanosohbet_smoke
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/ari/users/ytanis/.cache/nanochat
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
PY=/ari/users/ytanis/envs/nanochat/bin
cd /ari/users/ytanis/nanosohbet-repo

echo "Dugum: $(hostname)  Baslangic: $(date)"

# kucuk model, 30 adim: veri+tokenizer+egitim yolunun ucdan uca dogrulanmasi
$PY/torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=8 \
    --device-batch-size=8 \
    --num-iterations=30 \
    --core-metric-every=-1 \
    --sample-every=25 \
    --eval-every=25 \
    --eval-tokens=524288 \
    --save-every=-1 \
    --run=dummy

echo "SMOKE_TEST_OK  Bitis: $(date)"
