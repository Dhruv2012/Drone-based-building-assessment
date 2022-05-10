# Estimation of Window and Storey Parameters 
**NOTE:** The training script and model architecture have been taken from the [this repo](https://github.com/lck1201/win_det_heatmaps). More details about their paper [here](http://jcst.ict.ac.cn/EN/10.1007/s11390-020-0253-4). If you use the zju_facade dataset or their training script/models please cite them.

This dir contains training/testing/inference scripts for window detection, Template Matching-based post_processing module and vertical plane mapping module.

# Directory Structure:

## Directories

**sample_seq_data**: 3 sample vertical sequences of a building. Contains images, log file and CSV file(window coordinates obtained from post_processing module). \
**post_processing**: Entire Post-processing module based on template matching and Non-maxima suppression(NMS). Detects windows through template matching where the model fails and removes multiple instances through NMS. This is called during model inference (infer.py) and a CSV file containing window coordinates is generated.\
**mapToVerticalPlane**: Contains python class to map windows to a dummy vertical Plane. Also contains scripts and jupyter notebook on running it.\
**templateMatching**: Contains jupyter nb of Template Matching module showing all intermediate results. \
**recordedWindowCoordinates**: CSV Files(containing window coordinates) of different sequences generated from post-processing module.

# Pre-Trained models
Models trained on zju_facade and our IIIT-H dataset can be found [here](https://drive.google.com/drive/folders/1untOz0j8zHKBELrTC69r2wu8YpX_TYth?usp=sharing).


# Preparation
**Environment**

Please install PyTorch following the [official webite](https://pytorch.org/). In addition, you have to install other necessary dependencies.
```bash
pip3 install -r requirements.txt
```
# Usage
**Train**
```bash 
python train.py --cfg /path/to/yaml/config \
    --data /path/to/data/root \
    --out /path/to/output/root
```

**Test**
```bash 
python test.py --cfg /path/to/yaml/config --model /path/to/model \
    --data /path/to/data/root \
    --out /path/to/output/root
```


**Inference**
```bash
python infer.py --cfg /path/to/yaml/config \
                --model /path/to/model \
                --infer /path/to/image/directory
```
-> This generates 2 dirs: **infer_result** and **post_process_result**. \
**infer_result**: Results from model inference. \
**post_process_result**: Results from post-processing module. We also output a CSV file containing window coordinates of all windows(of each image) of a sequence. 
