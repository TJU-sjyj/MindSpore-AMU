from mindspore import load_checkpoint
import os
import random
import argparse
import yaml
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models as torchvision_models
import mindspore
from datasets.imagenet_lzy import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from backbones.load_model import load_model
from backbones.heads import *

def load_features(args, split, model, loader, tfm_norm, model_name, backbone_name):

    """features, labels = [], []
    middle_features = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            if hasattr(model, 'encode_image') and callable(getattr(model, 'encode_image')):
                image_features = model.encode_image(tfm_norm(images)) # for clip model
            else:
                image_features = model(tfm_norm(images))
            features.append(image_features.cpu()) #
            labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)
    features = features.cuda() 
    torch.save(features, args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_f.pt")
    torch.save(labels, args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_l.pt")
    if len(middle_features) > 0:
        middle_features = torch.cat(middle_features)
        torch.save(middle_features, args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_middle_f.pt")"""

    features = torch.load(args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_f.pt")
    labels = torch.load(args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_l.pt")
    middle_features = None

    return features, labels, middle_features

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def main():
    from parse_args import parse_args
    # Load config file
    parser = parse_args()
    args = parser.parse_args()
    print("\nTest progress.")
    cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir
    # CLIP
    clip_model, preprocess = clip.load(args.clip_backbone)
    clip_model.eval()
    # AUX MODEL 
    aux_model, tfm_aux, args.feat_dim = load_model(args.aux_model_name, args.aux_backbone)        
    aux_model.cuda()
    aux_model.eval()  

    # ImageNet dataset
    random.seed(args.rand_seed)
    torch.manual_seed(args.torch_rand_seed)
    print("Preparing ImageNet dataset.")
    #imagenet = ImageNet(args.root_path, args.shots,preprocess)
    imagenet = ImageNet(args.root_path, args.shots)
    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=128, num_workers=8, shuffle=False)
    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = gpt_clip_classifier(imagenet.classnames, None, clip_model, imagenet.template)
    
    print("\nLoading visual features and labels from test set.")


    print("\nLoading CLIP feature.")
    test_clip_features, test_labels, _= load_features(args, "test", clip_model, test_loader, tfm_norm=tfm_clip, model_name='clip',backbone_name=args.clip_backbone)
    
    print(f"\nLoading AUX feature ({args.aux_model_name}).")
    test_aux_features, test_labels, _= load_features(args, "test", aux_model, test_loader, tfm_norm=tfm_aux, model_name=args.aux_model_name, backbone_name=args.aux_backbone)
    
    test_clip_features, test_aux_features = test_clip_features.cuda(), test_aux_features.cuda()


    if args.checkpoint_version == "pytorch":
        print("loading pytorch checkpoint.")
        adapter = Linear_Adapter(args.feat_dim, 1000).cuda()
        adapter.weight = torch.load('/home/sjyjxz/mindaspore/AMU1/caches/ImageNet/best_mocov3_resnet50_adapter_1shots.pt')
    elif args.checkpoint_version == "mindspore":  
        print("loading mindspore checkpoint.")
        adapter = mindspore.nn.Dense(args.feat_dim,1000)
        param_dict = load_checkpoint(args.checkpoint_path)
        load_param_into_net(adapter, param_dict)

    
    clip_logits = 100. * test_clip_features @ clip_weights
    clip_acc = cls_acc(clip_logits, test_labels)
    print(f"CLIP Acc: {clip_acc}%")

    aux_logits = adapter(test_aux_features)
    tip_logits = clip_logits + aux_logits * 2
    # 计算当前alpha值对应的准确率
    acc = cls_acc(tip_logits, test_labels)
    print(f"AMU Acc: {acc}%")

    #print("**** AMU test accuracy: {:.2f}. ****\n".format(acc))
    
if __name__ == '__main__':
    main()