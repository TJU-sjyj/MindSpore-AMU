# AMU-Tuning: Effective Logit Bias for CLIP-based Few-shot Learning

This is an code implementation base on Mindspore2.2 and pytorch 1.7.1  of CVPR2024 paper (AMU-Tuning: Effective Logit Bias for CLIP-based Few-shot Learning), created by Qilong Wang, Zhenyi Lin and Yuwei Tang.

## Introduction
This paper proposes a novel AMU-Tuning method to learn effective logit bias for CLIP-based few shot classification. Specifically, our AMU-Tuning predicts logit bias by exploiting the appropriate Auxiliary features, which are fed into an efficient feature-initialized linear classifier with Multi-branch training. Finally, an Uncertainty based fusion is developed to incorporate logit bias into CLIP for few-shot classification. The experiments are conducted on several widely used benchmarks, and the results show AMU-Tuning clearly outperforms its counterparts while achieving state-of-the-art performance of CLIP based few-shot learning without bells and whistles.
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

### Data preparation
Download and extract ImageNet train and val images from http://image-net.org/. (https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)

Our dataset is set up to primarily follow [CaFo](https://github.com/OpenGVLab/CaFo).
### Evaluation
To evaluate a pre-trained model on ImageNet val with GPUs run:

```
CUDA_VISIBLE_DEVICES={device_ids}  python test.py --checkpoint_version={CHECKPOINT_VISION} --checkpoint_path={CHECKPOINT_PATH}
```
### Training

#### Train with ResNet

We provide a script file where the specific parameters can be referenced in [parse_args.py](https://github.com/TJU-sjyj/MindSpore-AMU/parse_args.py)

Run the training script:

```bash
sh run.sh
```
## Acknowledgement
This repo benefits from [Tip](https://github.com/gaopengcuhk/Tip-Adapter) and [CaFo](https://github.com/OpenGVLab/CaFo). Thanks for their works.



## Contact
If you have any questions or suggestions, please feel free to contact us: .
