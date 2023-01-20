#!/bin/bash
#SBATCH --job-name=submitter
#SBATCH --account=gts-smaguluri3
#SBATCH -t1:12:00
#SBATCH --output=submitter.txt
#SBATCH -q inferno
#SBATCH -o outputLogs/submitter.txt
#SBATCH --mail-user=meinhorn6@gatech.edu

cd $SLURM_SUBMIT_DIR
module load anaconda3/2022.05
conda activate torch
srun python submit.py tExp.sh