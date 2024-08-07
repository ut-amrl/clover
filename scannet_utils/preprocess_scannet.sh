#!/usr/bin/env bash

CURR_DIR=$(realpath $(dirname $0))
scenes=(scene0000_00)

DATA_DIR=data/ScanNet

for scene in "${scenes[@]}"; do
    python $CURR_DIR/preprocess_scene.py --root_dir data/ScanNet --scene_id $scene
done