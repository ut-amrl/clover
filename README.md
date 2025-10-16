<h3 align="center"><strong>CLOVER: Context-Aware Long-Term Object Viewpoint- and Environment- Invariant Representation Learning</strong></h3>

  <p align="center">
    <a href="https://mandi1267.github.io">Amanda Adkins*</a>,
    <a href="">Dongmyeong Lee*</a>,
    <a href="https://www.joydeepb.com">Joydeep Biswas</a>
    <br>
    *Equal Contribution.
    <br>
    The University of Texas at Austin
    <br>
    <b>IEEE Robotics and Automation Letters, 2025</b>

</p>

<div align="center">
 <a href='https://arxiv.org/abs/2407.09718'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <!-- <a href='TODO website'><img src='https://img.shields.io/badge/Project-Page-orange'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -->
 <a href='https://youtu.be/Pz_u3ORTIWc'><img src='https://img.shields.io/badge/YouTube-Video-yellow'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://github.com/ut-amrl/clover/tree/master?tab=MIT-1-ov-file'><img src='https://img.shields.io/badge/License-MIT-green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <br>
 <br>
</div>



Thank you for your interest in CLOVER. 


## CODa-ReID Dataset Download
You need (1) images from CODa and (2) our annotations.
1. Images: Download from https://amrl.cs.utexas.edu/coda/download.html
2. Annotations (CODa-ReID): Download from https://doi.org/10.18738/T8/E9WFTW

**Dataset layout**
```
CODa-ReID
├── 2d_raw
│   ├── cam0
│   └── cam1
├── 3d_bbox
│   └── global
│       ├── Tree.json
|       ├── ...
└── annotations
    ├── cam0
    └── cam1
```



## Dataset Usage


## Requirements / Installation


## CLOVER Usage

CLOVER training and evaluation requires a config file that specifies model parameters and dataset details. 
For CODa Re-ID, this is train.yaml


### Training

From the `src` directory, run ```python train.py --config-name <config file name>```


### Inference
From the `src` directory, run ```python eval.py --config-name <config file name> ++ckpt=<path to checkpoint>```

## MapCLOVER

### Map Generation


### Inference

