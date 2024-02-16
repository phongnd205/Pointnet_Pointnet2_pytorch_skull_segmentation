# Pointnet_Pointnet2_pytorch_skull_segmentation
This is repository for training and predicting segmentation for skull part segmentation
# Pytorch Implementation of PointNet and PointNet++ 

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.

## Update
**2024/02/16:** 
(1) Adjust segmentation code for skull segmentation 

## Install
The latest codes are tested on Ubuntu 22.04, CUDA12.1, PyTorch 2.1.1 and Python 3.8:


## Classification (ModelNet10/40)
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

## Part Segmentation (ShapeNet)
### Data Preparation
Point cloud (3, m) with corresponding label (m) for each point
### Run
(1) For training: main.py
(2) For predicting:  
### Performance
| Model | Inctance avg Acc| avg IoU 
|--|--|--|	
|PointNet2 (Pytorch)|95.65	|91.67	
