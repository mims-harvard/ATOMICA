#!/bin/bash
########## Instruction ##########
# This script takes three optional environment variables:
# GPU / ADDR / PORT
# e.g. Use gpu 0, 1 and 4 for training, set distributed training
# master address and port to localhost:9901, the command is as follows:
#
# GPU="0,1,4" ADDR=localhost PORT=9901 bash train.sh
#
# Default value: GPU=-1 (use cpu only), ADDR=localhost, PORT=9901
# Note that if your want to run multiple distributed training tasks,
# either the addresses or ports should be different between
# each pair of tasks.
######### end of instruction ##########


########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/../..`
echo "Locate the project folder at ${CODE_DIR}"

######### setup python path ##########
PYTHON="/n/home13/afang/.conda/envs/interactenv1/bin/python"


########## parsing JSON configs ##########
if [ -z $1 ]; then
    echo "Config missing. Usage example: GPU=0,1 bash $0 <config>"
    exit 1;
fi
CONFIG=`cat $1 | $PYTHON -c "import sys, json; config = json.load(sys.stdin); print(\" \".join([f'--{key}' + (f' {config[key]}' if type(config[key])!=bool else '') for key in config if config[key]]))"`

########## setup  distributed training ##########
GPU="${GPU:--1}" # default using CPU
MASTER_ADDR="${ADDR:-localhost}"
MASTER_PORT="${PORT:-9901}"
echo "Using GPUs: $GPU"
echo "Master address: ${MASTER_ADDR}, Master port: ${MASTER_PORT}"

export CUDA_VISIBLE_DEVICES=$GPU
GPU_ARR=(`echo $GPU | tr ',' ' '`)

if [ ${#GPU_ARR[@]} -gt 1 ]; then
    export OMP_NUM_THREADS=2
	PREFIX="torchrun --nproc_per_node=${#GPU_ARR[@]} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --nnodes=1"
else
    PREFIX="$PYTHON"
fi


########## start training ##########
cd $CODE_DIR
${PREFIX} train.py --gpu "${!GPU_ARR[@]}" ${CONFIG}