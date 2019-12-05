#!/usr/bin/env bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0
source activate C4
python ./test/test_sq_1stage.py  --pth_path0 './trained_models/C4_sq_1stage/fold0.pth' --pth_path1 './trained_models/C4_sq_1stage/fold1.pth' --pth_path2 './trained_models/C4_sq_1stage/fold2.pth'
