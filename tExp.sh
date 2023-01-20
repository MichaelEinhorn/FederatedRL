#!/bin/bash
#SBATCH -J p#
#SBATCH -A gts-smaguluri3
#SBATCH -N1 -nf#    # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=1G            # Memory per core
#SBATCH -t 0-3:00:00             		
#SBATCH -q embers
#SBATCH -o ~/scratch/logs/p#.txt
#SBATCH --mail-user=meinhorn6@gatech.edu
#SBATCH --mail-type=FAIL

cd $SLURM_SUBMIT_DIR
module load anaconda3/2022.05
conda activate torch
srun python test.py --trial t# --syncBackups sb# --fedP f# --alpha a# --epsilon e#
