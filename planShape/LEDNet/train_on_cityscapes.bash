#!/bin/bash
#SBATCH -A rrc
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH -w gnode07
#SBATCH --time=4-00:00:00
#SBATCH --output=trainLEDNet_cityscapes_logs.txt

# echo "Copying CityScapes dataset to scratch of computeNode"
# rsync -zaP dhruv.r.patel14@ada.iiit.ac.in:/share1/dataset/cityscapes/cityscapes.tar /scratch/dhruv.r.patel14/datasets
# echo "Copy done successfully!!"

echo "Extracting cityscapes"
tar -xvf /scratch/dhruv.r.patel14/datasets/cityscapes.tar