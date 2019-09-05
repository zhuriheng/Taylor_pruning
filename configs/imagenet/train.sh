#!/usr/bin/env bash
set -eux
set -o pipefail

source activate torch1.1py3.5

cd ../../

nohup python -u main.py --name=runs/resnet50/resnet50_prune72 \
                --dataset=Imagenet \
                --data=/workspace/mnt/cache/ImageNet-pytorch \
                --lr=0.001 \
                --lr-decay-every=10 \
                --momentum=0.9 \
                --epochs=25 \
                --batch-size=256 \
                --pruning=True \
                --seed=0 \
                --model=resnet50 \
                --load_model=./models/pretrained/resnet50-19c8e357.pth \
                --mgpu=True \
                --group_wd_coeff=1e-8 \
                --wd=0.0 \
                --tensorboard=True \
                --pruning-method=22 \
                --no_grad_clip=True \
                --pruning_config=./configs/imagenet_resnet50_prune72.json \
                > imagenet_resnet50_resnet101_prune72_v0.1.log 2>&1 &
