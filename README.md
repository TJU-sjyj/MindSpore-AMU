# AMU-Tuning: Effective Logit Bias for CLIP-based Few-shot Learning

This is an code implementation base on Mindspore2.2 and pytorch 1.7.1  of ***CVPR 2024*** paper [**AMU-Tuning: Effective Logit Bias for CLIP-based Few-shot Learning**](https://arxiv.org/abs/2404.08958)

## Introduction
This paper proposes a novel **AMU-Tuning** method to learn effective logit bias for CLIP-based few shot classification. Specifically, our AMU-Tuning predicts logit bias by exploiting the appropriate ***A***uxiliary features, which are fed into an efficient feature-initialized linear classifier with ***M***ulti-branch training. Finally, an ***U***ncertainty based fusion is developed to incorporate logit bias into CLIP for few-shot classification. The experiments are conducted on several widely used benchmarks, and the results show AMU-Tuning clearly outperforms its counterparts while achieving state-of-the-art performance of CLIP based few-shot learning without bells and whistles.
<div align="center">
  <img src="framework.png"/>
</div>

## Usage

### Environments
●OS：16.04  
●CUDA：11.6  
●Toolkit：MindSpore2.2 & PyTorch 1.7.1  
●GPU:GTX 3090 

## Requirements
### Install

create virtual enviroment and install dependencies:

```bash
git clone https://github.com/TJU-sjyj/MindSpore-AMU
conda create -n AMU python=3.7
conda activate AMU
# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

CUDA 10.1 
```bash
conda install mindspore-gpu cudatoolkit=10.1 -c mindspore -c conda-forge
```
CUDA 11.1 
```bash
conda install mindspore-gpu cudatoolkit=11.1 -c mindspore -c conda-forge
```
validataion 
```bash
python -c "import mindspore;mindspore.run_check()"
```

### Dataset
Our dataset setup is primarily based on Tip-Adapter. Please follow [DATASET.md](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md) to download official ImageNet and other 10 datasets.

### Foundation Models
* The pre-tained weights of **CLIP** will be automatically downloaded by running.
* The pre-tained weights of **MoCo-v3** can be download at [MoCo v3](https://github.com/facebookresearch/moco-v3).

## Get Started

### One-Line Command by Using `run.sh`

We provide `run.sh` with which you can complete the pre-training + fine-tuning experiment cycle in an one-line command.

### Arguments
- `clip_backbone` is the name of the backbone network of CLIP visual coders that will be used (e.g. RN50, RN101, ViT-B/16).  
- `lr` learning rate for adapter training.  
- `shots` number of samples per class used for training.  
- `alpha` is used to control the effect of logit bias.  
- `lambda_merge`  is a hyper-parameter in **Multi-branch Training**  

More Arguments can be referenced in [parse_args.py](https://github.com/TJU-sjyj/AMU-Tuning/parse_args.py)

### Training Example
You can use this command to train a **AMU adapter** with ViT-B-16 as CLIP's image encoder by 16-shot setting for 50 epochs.  

```bash
CUDA_VISIBLE_DEVICES=0 python train.py\
    --rand_seed 2 \
    --torch_rand_seed 1\
    --exp_name test_16_shot  \
    --clip_backbone "ViT-B-16" \
    --augment_epoch 1 \
    --init_alpha 0.5\
    --lambda_merge 0.35\
    --train_epoch 50\
    --lr 1e-3\
    --batch_size 8\
    --shots 16\
    --root_path 'your root path' \
```
### Test Pretrained Model
You can use the test scripts [test.sh](https://github.com/TJU-sjyj/MindSpore-AMU/blob/main/test.sh) to test the pretrained model. More Arguments can be referenced in [parse_args.py](https://github.com/TJU-sjyj/AMU-Tuning/parse_args.py)

## Main Results
|Method           | Acc-MindSpore  | Acc-PyTorch   | Checkpoint(PyTorch)|Checkpoint(MindSpore)|
| ----------------| -------------- | ------- | ----------------- |---|
| MoCov3-ResNet50-16shot-lmageNet1k  |69.98|70.02|[Download](https://drive.google.com/file/d/1U8V8aho9ne5Diz0swK9m_-1cEElflnMg/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1FzvOUxfwv8MP3qpWJuRhYgJZQ-hGrcHh/view?usp=drive_link)|
| MoCov3-ResNet50-8shot-lmageNet1k   |68.21|68.25|[Download](https://drive.google.com/file/d/1QmF3EoaxIohMzQHj6P7fmvK73ab0lfBP/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1ICSjE5DfwfkTlV0b72XsYOVaj3Ej6XxB/view?usp=drive_link)|
| MoCov3-ResNet50-4shot-lmageNet1k   |65.79|65.92|[Download](https://drive.google.com/file/d/1ad0-yVsN3JV_-7mbFlYq1mpM0XoviIUs/view?usp=drive_link)|[Download](https://drive.google.com/file/d/14CjUDy4-PMYsjSNpjn8NAHIu7FNm5TXt/view?usp=drive_link)|
| MoCov3-ResNet50-2shot-lmageNet1k   |64.19|64.25|[Download](https://drive.google.com/file/d/15R9ohx9z2VsC0h7Z1gAeidWDy3bfHByu/view?usp=drive_link)|[Download](https://drive.google.com/file/d/18sMZ9ndF9rAGDD7XhtDvdY7Kqvp03SDi/view?usp=drive_link)|
| MoCov3-ResNet50-1shot-lmageNet1k   |62.57|62.60|[Download](https://drive.google.com/file/d/19fEVFa241C4VZ88k2d8uVEAvkAuITzfI/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1g-QoP1XB6JQkYV2av41IJ8IDoGUyuiPd/view?usp=drive_link)|



## Acknowledgement
This repo benefits from [Tip](https://github.com/gaopengcuhk/Tip-Adapter) and [CaFo](https://github.com/OpenGVLab/CaFo). Thanks for their works.



## Contact
If you have any questions or suggestions, please feel free to contact us: tangyuwei@tju.edu.cn and linzhenyi@tju.edu.cn.
