#!/usr/bin/env bash
set -x
set -e

source activate C4
python ./data/img2npy.py
