#!/usr/bin/env bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=6
foldnum=0
env=C4_sq_3stage_fold$foldnum.`date +'%Y-%m-%d_%H-%M-%S'`
source activate C4
python ./train/train_sq_3stage.py --nepoch 2000 --alpha1 0.33 --alpha2 0.33 --foldnum $foldnum --env $env --pth_path './trained_models/C4_sq_1stage/fold'$foldnum'.pth'

