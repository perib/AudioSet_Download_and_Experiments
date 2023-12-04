#!/bin/bash
#SBATCH --job-name=uBIGDownloading_YT_Wavs
#SBATCH --output=./output/unbalancedset/download%j.out
#SBATCH --error=./output/unbalancedset/download%j.err
#SBATCH --array=0-24
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=48:00:00

numJobs=25
numProcesses=10
Filename="unbalanced_train_segments.csv"
audioQuality="44k"
SectionNum=$SLURM_ARRAY_TASK_ID

python audiosetDL_python.py -a $Filename -b $audioQuality -c $SLURM_ARRAY_TASK_ID -d $numJobs -e $numProcesses -m "TRUE"
