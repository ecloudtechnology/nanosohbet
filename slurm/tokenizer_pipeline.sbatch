#!/bin/bash
#SBATCH --job-name=nanosohbet_tok
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out

set -e
export HF_HUB_OFFLINE=1
PY=/ari/users/ytanis/envs/torch/bin/python

echo "Dugum: $(hostname)  Baslangic: $(date)"
$PY ~/nanosohbet/scripts/train_tokenizer.py
$PY ~/nanosohbet/scripts/benchmark_tokenizers.py
echo "Bitis: $(date)"
