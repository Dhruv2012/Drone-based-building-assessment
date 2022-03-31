#!/bin/bash
#SBATCH -A rrc
#SBATCH -n 18
#SBATCH -w gnode07
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=trainLEDNet_logs.txt

dataset_dir=/scratch/dhruv.r.patel14/training_data/train_lednet
pretrainedEnc=/scratch/dhruv.r.patel14/training_data/train_lednet_enc_imgnet/model_best.pth.tar
trainedEnc=/home2/dhruv.r.patel14/LEDNet/save/train_lednet_pretrainedEnc-1/logs/model_encoder_best.pth

# python main.py --savedir ./train_lednet_pretrainedEnc-1/logs --datadir $dataset_dir --num-epochs 500 --batch-size 8 --num-workers 16 --epochs-save 5 --pretrainedEncoder $pretrainedEnc

python main.py --savedir ./train_lednet_pretrainedEnc-1/logs --datadir $dataset_dir --num-epochs 400 --batch-size 8 --num-workers 16 --epochs-save 5 --state '../save/train_lednet_pretrainedEnc-1/logs/model_best.pth.tar'

# python main.py --savedir ./train_lednet_pretrainedEnc-1/logs --datadir $dataset_dir --num-epochs 800 --batch-size 8 --num-workers 16 --epochs-save 5  --pretrainedEncoder $trainedEnc --decoder

# python main.py --savedir ./train_lednet_scratch/logs --datadir $dataset_dir --num-epochs 50 --batch-size 8 --num-workers 16 --epochs-save 5 --state '../save/train_lednet+/logs/model_best_enc.pth.tar' --decoder
