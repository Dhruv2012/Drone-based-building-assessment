# Estimation of Window and Storey Parameters 
**NOTE:** The training script and models have been taken from the [this repo](https://github.com/lck1201/win_det_heatmaps). More details about their paper [here](http://jcst.ict.ac.cn/EN/10.1007/s11390-020-0253-4). If you use the zju_facade dataset or their training script/models please cite them.

It contains training/testing/inference scripts for window detection, Template Matching-based post_processing module and vertical plane mapping module.

# Directory Structure:

## Directories

**sample_seq_data**: 3 sample vertical sequences of a building. Contains images, log file and CSV file(window coordinates obtained from post_processing module). \
**post_processing**: Entire Post-processing module based on template matching and Non-maxima suppression(NMS). Detects windows through template matching where the model fails and removes multiple instances through NMS. \
**templateMatching**: Contains jupyter nb of Template Matching module showing all intermediate results. \
**recordedWindowCoordinates**: CSV Files(containing window coordinates) of different sequences generated from post-processing module. \
**models**: Pre-trained models on zju_facade and IIIT-H dataset.


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

 
# Acknowledgement
The major contributors of this repository include [Chuankang Li](https://github.com/lck1201), [Yuanqing Zhang](https://github.com/yuanqing-zhang), [Shanchen Zou](https://github.com/Generior), and [Hongxin Zhang](https://person.zju.edu.cn/zhx).

