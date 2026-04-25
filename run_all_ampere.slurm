#!/bin/bash
#SBATCH --job-name=jusdef-all
#SBATCH --partition=gpu          # TODO: set to your GPU partition name
#SBATCH --gres=gpu:a100:1        # or gpu:ampere:1 depending on cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_all-%j.out
#SBATCH --error=logs/run_all-%j.err

cd ~/Jusdef
./scripts/run_all.sh
