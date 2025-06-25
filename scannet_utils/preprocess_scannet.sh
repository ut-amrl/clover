#!/usr/bin/env bash

PROJECT_DIR=$(realpath $(dirname $0)/..)

DATA_DIR=$PROJECT_DIR/data/ScanNet
scenes=$(basename -a $(ls -d "$DATA_DIR/scans"/*/))

for scene in $scenes; do
    if [[ ! $scene == *_00 ]]; then
        echo "Skipping scene $scene because it does not end with _00."
        continue
    fi

    # check if the scene has been preprocessed
#     if [ -f "$DATA_DIR/2d_instance/${scene}.json" ]; then
#         echo "Scene $scene has been preprocessed. Skip."
#         continue
#     fi
    CUDA_VISIBLE_DEVICES=6 python $PROJECT_DIR/scannet_utils/preprocess_scene.py --root_dir data/ScanNet --scene_id $scene
done
