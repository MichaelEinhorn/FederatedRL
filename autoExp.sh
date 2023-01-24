#!/bin/bash
#SBATCH -Jmdpv4-t0-sb10000-f16-a01-e1
#SBATCH -Agts-smaguluri3
#SBATCH -N1 -n16    # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=1G            # Memory per core
#SBATCH -t3:00:00             		
#SBATCH -qembers
#SBATCH -o/storage/home/hcoda1/2/meinhorn6/scratch/logs/mdpv4-t0-sb10000-f16-a01-e1-%j.out
#SBATCH --mail-user=meinhorn6@gatech.edu
#SBATCH --mail-type=FAIL
cd $HOME/p-smaguluri3-0/RL

module load anaconda3/2022.05
conda activate torch
python test.py --prefix mdpv4-t0-sb10000-f16-a01-e1 --trial 0 --syncBackups 10000 --fedP 16 --alpha 0.1 --epsilon 1
