#!/bin/bash
# conda init bash
# This script is used to estimate the roof layout of a building
LOG="./RoofLayoutEstimation/log.txt"
exec > >(tee $LOG)
echo "Downloading the dataset"
sleep 2s
# gdown 1v3qJkcBVAczAL5Sdb2HebaW_Thy90bcq
echo "Done!"

echo "Extracting images from the video"
sleep 10s
# rm -rf images
# python VideoToImage.py --video DJI_0641.MP4 --savedir images
echo "Done!"
touch ./RoofLayoutEstimation/out.log
echo "Estimating the roof masks"
sleep 20s
cd LEDNet/test
chmod 777 test.py
# rm -rf RoofMasks
# python test.py --datadir ../../images --resultdir ../../RoofMasks >> ../../RoofLayoutEstimation/out.log
cd ../../
echo "Done!"

echo "Displaying Roof Mask Results"
chmod 777 saveroofmaskresults.py
python saveroofmaskresults.py -i images -r RoofMasks >> ./RoofLayoutEstimation/out.log
echo "Done!"

echo "Estimating the NSE Masks"
# rm -rf ObjectMasks
# mkdir ObjectMasks
cd Detic
chmod 777 demo.py
sleep 20s
# python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ../images/*.png --output ../ObjectMasks --vocabulary custom --custom_vocabulary solar_array,air_conditioner,vent,box,sink --confidence-threshold 0.5 --opts MODEL.WEIGHTS Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth >> ../RoofLayoutEstimation/out.log
cd ..
echo "Done!"

echo "Displaying Intermediate Results before stitching"
chmod 777 generateintermediateresults.py
python generateintermediateresults.py -i images -r RoofMasks -o ObjectMasks >> ./RoofLayoutEstimation/out.log
echo "Done!"

touch ./RoofLayoutEstimation/final_results/final_results.txt