#!/bin/bash
#SBATCH --job-name=submitter
#SBATCH --account=gts-smaguluri3
#SBATCH -N1 --ntasks-per-node=1                 # Number of nodes and cores required
#SBATCH --mem-per-cpu=1G                        # Memory per core
#SBATCH -t1-12:00:00
#SBATCH -qinferno
#SBATCH -o/storage/home/hcoda1/2/meinhorn6/scratch/logs/Submitter-%j.out
#SBATCH --mail-user=meinhorn6@gatech.edu
#SBATCH --mail-type=ALL
cd $HOME/p-smaguluri3-0/RL

module load anaconda3/2022.05
conda activate torch
python submit.py tExp.sh
