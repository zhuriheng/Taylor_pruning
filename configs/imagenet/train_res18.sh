#!/usr/bin/env bash
set -eux
set -o pipefail

source activate torch1.1py3.5

cd ../../

nohup python -u main.py --name=runs/resnet18/resnet18_prune75 \
                --dataset=Imagenet \
                --data=/workspace/mnt/cache/ImageNet-pytorch \
                --lr=0.001 \
                --lr-decay-every=20 \
                --momentum=0.9 \
                --epochs=70 \
                --batch-size=512 \
                --pruning=True \
                --seed=0 \
                --model=resnet18 \
                --load_model=./models/pretrained/resnet18-5c106cde.pth \
                --mgpu=True \
                --group_wd_coeff=1e-8 \
                --wd=0.0 \
                --tensorboard=True \
                --pruning-method=22 \
                --no_grad_clip=True \
                --pruning_config=./configs/imagenet_resnet18_prune75.json \
                > imagenet_resnet18_prune75_v0.1.log 2>&1 &
