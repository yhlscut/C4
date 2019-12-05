#!/usr/bin/env bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=6
foldnum=0
env=squeezenet_1stage_fold$foldnum.`date +'%Y-%m-%d_%H-%M-%S'`
source activate C4
python ./train/train_sq_1stage.py --nepoch 2000 --foldnum $foldnum --env $env

