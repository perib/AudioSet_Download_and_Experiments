#!/bin/bash
#SBATCH --job-name=wavTOCoch
#SBATCH --output=./output/ttococh5%j.out
#SBATCH --error=./output/ttococh5%j.err
#SBATCH --gres=gpu:titan-x:1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=24:00:00

module unload openmind/cuda/7.5
export LD_LIBRARY_PATH=/cm/shared/apps/gcc/4.8.4/lib:/cm/shared/apps/gcc/4.8.4/lib64:/cm/shared/openmind/cuda/8.0/extras/CUPTI/lib64

module load openmind/cuda/8.0

module load openmind/cudnn/8.0-5.1
# source activate tensorflow
source activate /home/ribeirop/Pedro-env
unset XDG_RUNTIME_DIR

hdf5file="/home/ribeirop/OMFOLDER/audiosetDL/eval_segments.hdf5"
Outputpath="/om/user/ribeirop/audiosetDL/eval_stripped/"

python WavHDF5_to_CochHDF5_python.py -o $Outputpath -w $hdf5file
