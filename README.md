# GaussianTalker: Real-Time High-Fidelity Talking Head Synthesis with Audio-Driven 3D Gaussian Splatting
<br>

This is our official implementation of the paper 

"GaussianTalker: Real-Time High-Fidelity Talking Head Synthesis with Audio-Driven 3D Gaussian Splatting"

## Introduction
![image](./docs/structure.png)
<!-- <br> -->

For more information, please check out our [Paper](dir) and our [Project page](dir).

## Installation
We implemented & tested **GaussianTalker** with NVIDIA RTX 3090 and A6000 GPU.

Run the below codes for the environment setting. ( details are in requirements.txt )
```bash
git clone https://github.com/joungbinlee/GaussianTalker.git
cd GaussianTalker
git submodule update --init --recursive
conda create -n GaussianTalker python=3.7 
conda activate GaussianTalker

pip install -r requirements.txt
pip install -e submodules/custom-bg-depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

## Data Preparation

```
├── data
│   | (your dataset name)
│       ├── gt_imgs
│           ├── 0.jpg
│           ├── 1.jgp
│           ├── 2.jgp
│           ├── ...
│       ├── ori_imgs
│           ├── 0.jpg
│           ├── 0.lms
│           ├── 1.jgp
│           ├── 1.lms
│           ├── ...
│       ├── parsing
│           ├── 0.png
│           ├── 1.png
│           ├── 2.png
│           ├── 3.png
│           ├── ...
│       ├── torso_imgs
│           ├── 0.png
│           ├── 1.png
│           ├── 2.png
│           ├── 3.png
│           ├── ...
│       ├── au.csv
│       ├── aud_ds.npy
│       ├── aud_novel.wav
│       ├── aud_train.wav
│       ├── aud.wav
│       ├── bc.jpg
│       ├── (your dataset name).mp3
│       ├── track_params.pt
│       ├── transforms_train.json
│       ├── transforms_val.json

