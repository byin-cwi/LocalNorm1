#!/bin/bash
#SBATCH --gres=gpu:1
python mnist_run.py --Norm_type batch
echo 'BatchNorm is finishied'
python mnist_run.py --Norm_type local
echo 'LocalNorm is finishied'
python mnist_run.py --Norm_type switch
echo 'SwitchNorm is finishied'
python visualize_result.py
