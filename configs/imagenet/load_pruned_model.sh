#!/usr/bin/env bash
set -eux
set -o pipefail

source activate torch1.1py3.5

cd ../../

nohup python -u pruned_network.py \
                --load_weights_path ./weights/best_model_42.weights \
                --model_arch resnet18 \
                --save_weight_path ./weights/pruned_network_res18.weights \
                > ./log/pruned_res18_network.log 2>&1 &