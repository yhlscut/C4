#!/usr/bin/env bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0
source activate C4
python ./test/test_sq_3stage.py --alpha1 0.33 --alpha2 0.33 --pth_path0 './trained_models/C4_sq_3stage/fold0.pth' --pth_path1 './trained_models/C4_sq_3stage/fold1.pth' --pth_path2 './trained_models/C4_sq_3stage/fold2.pth' 
