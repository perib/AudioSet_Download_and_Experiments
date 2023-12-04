#!/bin/bash
#SBATCH --job-name=uuwavTOhdf5
#SBATCH --output=./output/etohdf5%j.out
#SBATCH --error=./output/etohdf5%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=24:00:00

DownloadPath="/home/ribeirop/OMFOLDER/audiosetDL/unbalanced_train_segments_downloads"
Filename="/home/ribeirop/OMFOLDER/audiosetDL/unbalanced_train_segments.csv"
Outputpath="/om/user/ribeirop/audiosetDL/"

python Wav_into_HDF5_python.py -o $Outputpath -c $Filename -p $DownloadPath
