#!/usr/bin/env bash

PROJECT_DIR=$(realpath $(dirname $0)/..)

DATA_DIR=$PROJECT_DIR/data/ScanNet
scenes=$(basename -a $(ls -d "$DATA_DIR/scans"/*/))

for scene in $scenes; do
    python $PROJECT_DIR/scannet_utils/preprocess_scene.py --root_dir data/ScanNet --scene_id $scene
done