# DPGLA: Bridging the Gap between Synthetic and Real Data for Unsupervised Domain Adaptation in 3D LiDAR Semantic Segmentation
This is the Pytorch implementation of our DPGLA paper submitted in IROS 2025

## Installation
The code has been tested with Conda environment with Python 3.8, CUDA 10.2/11.1, pytorch 1.8.0, wandb, [Open3D](https://www.open3d.org/), [pytorch-lighting 1.4.1](https://lightning.ai/) and [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine). 


## Dataset
[SynLiDAR](https://github.com/xiaoaoran/SynLiDAR)
[SemanticKITTI](https://semantic-kitti.org/)
[SemanticPOSS](http://www.poss.pku.edu.cn/semanticposs.html)

## Usage
### Pre-training
```sh
python train_source.py --config_file configs/source/synlidar2semantickitti.yaml
```
### UDA
```sh
python adapt_cosmix_uda.py --config_file configs/adaptation/uda/synlidar2semantickitti_cosmix.yaml
```
### Test
```sh
python eval.py --config_file configs/config-file-of-the-experiment.yaml --resume_path PATH-TO-EXPERIMENT --is_student --eval_target --save --save_predictions
```

## Acknowledgment
This code is heavily borrowed from [CoSMix](https://github.com/saltoricristiano/cosmix-uda/tree/main?tab=readme-ov-file) and [LaserMix](https://github.com/ldkong1205/LaserMix)
