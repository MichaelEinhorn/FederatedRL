#!/bin/bash
#SBATCH -J submitter
#SBATCH -A gts-smaguluri3
#SBATCH -t 1-12:00
#SBATCH -q inferno
#SBATCH -o outputLogs/submitter.txt
#SBATCH --mail-user=meinhorn6@gatech.edu
#SBATCH --mail-type=FAIL

cd $SLURM_SUBMIT_DIR
module load anaconda3/2022.05
conda activate torch
srun python submit.py tExp.sh