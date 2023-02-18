#!/bin/bash
#SBATCH -Jp#
#SBATCH -Agts-smaguluri3
#SBATCH -N1 -nn#    # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=1G            # Memory per core
#SBATCH -t3:00:00             		
#SBATCH -qembers
#SBATCH -o/storage/home/hcoda1/2/meinhorn6/scratch/logs/p#-%j.out
#SBATCH --mail-user=meinhorn6@gatech.edu
#SBATCH --mail-type=FAIL
cd $HOME/p-smaguluri3-0/RL

module load anaconda3/2022.05
conda activate torch
python test.py --prefix p# --trial t# --syncBackups sb# --fedP f# --alpha a# --epsilon e#
