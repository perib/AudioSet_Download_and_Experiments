#!/bin/bash
#SBATCH --job-name=fzpredictcountsaudioset
#SBATCH --output=./output/fzpredictcountspntest%j.out
#SBATCH --error=./output/fzpredictcountspntest%j.err
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=30000
#SBATCH --time=150:00:00

module unload openmind/cuda/7.5
export LD_LIBRARY_PATH=/cm/shared/apps/gcc/4.8.4/lib:/cm/shared/apps/gcc/4.8.4/lib64:/cm/shared/openmind/cuda/8.0/extras/CUPTI/lib64

module load openmind/cuda/8.0

module load openmind/cudnn/8.0-5.1
# source activate tensorflow
source activate /home/ribeirop/Pedro-env
unset XDG_RUNTIME_DIR

learning_rate='0.00001'
name="predictcounts_5HP9_freeze"
task="PREDICT_COUNTS"
conv1filtersize='9'
poolmethod="HPOOL"
tbfolder="TB3"
musiconly="False"
conv1_times_hanning="False"
SAVE="True"
Freeze_Model_File="EPOCH_SAVED/Net_PREDICT_LABELS_A2UN91_5_HP9_nfull_lr1e-05_conv1FS9/model.ckpt-13"

python PedrosNetwork_python6.py -l $learning_rate -n $name -t $task -f $conv1filtersize -p $poolmethod -b $tbfolder -m $musiconly -x $conv1_times_hanning -s $SAVE -e $Freeze_Model_File
