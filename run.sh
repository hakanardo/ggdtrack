#!/usr/bin/env bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1 --mem-per-cpu=3100 -tasks-per-node=5

echo run.sh $*

source activate torch
python full_run.py $*
