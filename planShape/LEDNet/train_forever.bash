#!/bin/bash
#SBATCH -A research
#SBATCH -n 25
#SBATCH -w gnode16
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=trainLEDEnc_logs.txt

source ~/miniconda3/bin/activate
conda activate robo

echo "Copying ImageNet dataset to scratch of computeNode"
rsync -zaP dhruv.r.patel14@ada.iiit.ac.in:/share1/dataset/Imagenet2012 /scratch/dhruv.r.patel14
echo "Copy done successfully!!"

echo "Extracting dataset"
tar -xvf /scratch/dhruv.r.patel14/Imagenet2012/Imagenet-orig.tar && cp /home2/dhruv.r.patel14/LEDNet/extract_ImageNet.sh /scratch/dhruv.r.patel14/Imagenet2012/Imagenet-orig/
bash /scratch/dhruv.r.patel14/Imagenet2012/Imagenet-orig/extract_ImageNet.sh

save_dir=/scratch/dhruv.r.patel14/training_data/train_lednet_enc_imgnet
mkdir $save_dir
echo "Running Training script for training encoder on Imagenet" # Running training script
python /home2/dhruv.r.patel14/LEDNet/imagenet-pretrain/main.py /scratch/dhruv.r.patel14/Imagenet2012/Imagenet-orig -j 24 --start-epoch 6 --resume /scratch/dhruv.r.patel14/training_data/train_lednet_enc_imgnet/checkpoint.pth.tar --save_dir=$save_dir