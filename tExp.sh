#!/bin/bash
#SBATCH --job-name=p#
#SBATCH --account=gts-smaguluri3
#SBATCH -N1 --ntasks-per-node=f#    # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=1G            # Memory per core
#SBATCH -t3:00             		
#SBATCH -qembers
#SBATCH -oReport-outputLogs/p#.txt
#SBATCH --mail-user=meinhorn6@gatech.edu

cd $SLURM_SUBMIT_DIR
module load anaconda3/2022.05
conda activate torch
srun python test.py --trial t# --syncBackups sb# --fedP f# --alpha a# --epsilon e#
