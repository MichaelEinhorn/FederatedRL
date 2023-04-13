#!/bin/bash
#SBATCH -JimpalacoinrunNoLv
#SBATCH -Agts-smaguluri3
#SBATCH -N1 --gres=gpu:V100:1                       # Number of nodes and GPUs required
#SBATCH --mem-per-gpu=12G                           # Memory per gpu
#SBATCH -t6:00:00             		
#SBATCH -qinferno
#SBATCH -o/storage/home/hcoda1/2/meinhorn6/scratch/logs/impalacoinrun-%j.out
#SBATCH --mail-user=meinhorn6@gatech.edu
#SBATCH --mail-type=FAIL
cd $HOME/p-smaguluri3-0/RL

module load anaconda3/2022.05
conda activate torch
python train.py --model impalaVectorNoLv --epoch 1000 --num_models 1 --syncFreq 1
