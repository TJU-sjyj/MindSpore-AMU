# from https://github.com/facebookresearch/moco/blob/main/main_lincls.py

import torchvision.models as models
import torch
import torch.nn as nn
import os
from backbones.my_resnet import resnet50
#from my_resnet import resnet50
'''
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
'''

def load_moco(arch, pretrain_path):
    print("=> creating model '{}'".format(arch))
    #model = models.__dict__[arch]()
    model = resnet50()
    linear_keyword = 'fc' # for resnet
    if os.path.isfile(pretrain_path):
        print("=> loading checkpoint '{}'".format(pretrain_path))
        checkpoint = torch.load(pretrain_path, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint["state_dict"]
        # print(state_dict.keys())
        for k in list(state_dict.keys()):
        
            # for v1 & v2
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            
            # for v3
            elif k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {f"{linear_keyword}.weight", f"{linear_keyword}.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrain_path))
    else:
        print("=> no checkpoint found at '{}'".format(pretrain_path))
        raise FileNotFoundError
    
    model.fc = nn.Identity()
    
    return model

if __name__ == "__main__":
    
    model = load_moco("resnet50", "/home/sjyjxz/mindaspore/AMU/models/r-50-1000ep.pth.tar").cuda()
    #print(model)
    #print(model(torch.rand(32,3,224,224)).shape)
    #print(model.layer4[2])
    from copy import deepcopy
    x = deepcopy(model.layer4[2].conv1)
    #print(x)
    #统计参数量
    count = 0
    for name, param in x.named_parameters():
        count += param.numel()
    print(count)
    #print(model.feat_middle)
    
    
    
