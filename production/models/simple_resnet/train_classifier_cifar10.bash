#!/usr/bin/env bash

echo "Production Training Script for simple_resnet model on CIFAR-10"

MODEL_PATH="simple_resnet_cifar10.pth"

python train_classifier.py -e 30 -b 512 --n_rounds 1 --p_initial 1.0 -m $MODEL_PATH

python test_classifier.py -m $MODEL_PATH

echo "Model trained and tested; refer to the README.md for deployment"
