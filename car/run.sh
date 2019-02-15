#!/bin/bash
#SBATCH --gres=gpu:1

echo 'Downloading dataset ... '
cd Car-Recognition
wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz

echo 'Manage dataset'
python pre-process.py
echo 'Start training ...'

python train.py --Norm_type batch
echo 'BatchNorm is finishied'

python train.py --Norm_type local
echo 'LocalNorm is finishied'

# python test_on_noisy.py
