# GaussianTalker: Real-Time High-Fidelity Talking Head Synthesis with Audio-Driven 3D Gaussian Splatting
<a href="https://arxiv.org/abs/2404.16012"><img src="https://img.shields.io/badge/arXiv-2404.16012-%23B31B1B"></a>
<a href="https://ku-cvlab.github.io/GaussianTalker"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>
<br>

This is our official implementation of the paper 

"GaussianTalker: Real-Time High-Fidelity Talking Head Synthesis with Audio-Driven 3D Gaussian Splatting"

## Introduction
![image](./docs/structure.png)
<!-- <br> -->

For more information, please check out our [Paper](https://arxiv.org/abs/2404.16012) and our [Project page](https://ku-cvlab.github.io/GaussianTalker/).

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


## Download Dataset

We used talking portrait videos from [AD-NeRF](https://github.com/YudongGuo/AD-NeRF), [GeneFace](https://github.com/yerfor/GeneFace) and [HDTF dataset](https://github.com/MRzzm/HDTF). 
These are static videos whose average length are about 3~5 minutes.

You can see an example video with the below line:

```
wget https://github.com/yerfor/GeneFace/releases/download/v1.1.0/May.zip
```

We also used [SynObama](https://grail.cs.washington.edu/projects/AudioToObama/) for cross-driven setting inference.


## Data Preparation

[Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details) 

Put "01_MorphableModel.mat" to data_utils/face_tracking/3DMM/ 
    
```bash
cd data_utils/face_tracking
python convert_BFM.py
cd ../../
python data_utils/process.py ${YOUR_DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.mp4 
```

```
├── (your dataset dir)
│   | (dataset name)
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
│       ├── (dataset name).mp4
│       ├── track_params.pt
│       ├── transforms_train.json
│       ├── transforms_val.json
```

## Training


```bash
python train.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} --model_path ${YOUR_MODEL_DIR} --configs arguments/64_dim_1_transformer.py 
```


## Rendering

Please adjust the batch size to match your GPU settings.

```bash
python render.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} --model_path ${YOUR_MODEL_DIR} --configs arguments/64_dim_1_transformer.py --iteration 10000 --batch 128
```
    
