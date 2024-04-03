# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
generate ckpt from pth.
"""
import argparse
import torch
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint
from mindspore import load_checkpoint

parser = argparse.ArgumentParser(description="convert torch pretrain model to mindspore checkpoint.")
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--modality', type=str, default="RGB", choices=['RGB', 'Flow'])
parser.add_argument('--new_length', type=int, default=5)

modules = {'inception_3a', 'inception_3b', 'inception_3c',
           'inception_4a', 'inception_4b', 'inception_4c',
           'inception_4d', 'inception_4e', 'inception_5a', 'inception_5b'}

def print_weight_names_from_ckpt(ckpt_file):
    # 加载 .ckpt 文件
    param_dict = load_checkpoint(ckpt_file)
    
    # 打印所有权重的名称
    print("权重名称列表：")
    for name in param_dict:
        print(name)

def checkname(name):
    name = name.replace('.net', '')
    if 'bn' in name:
        name = name.replace('weight', 'gamma')
    if 'downsample.1' in name:
        name = name.replace('weight', 'gamma')
    if 'fc' in name:
        name = name.replace('module.', '')
    if 'layer' in name:
        name = name.replace('module.base_model.', '')
    else:
        name = name.replace('module.base_model', 'base_model')
    name = name.replace('running_mean', 'moving_mean')
    name = name.replace('running_var', 'moving_variance')
    if 'new_fc' not in name:
        name = name.replace('bias', 'beta')
    name = name.replace('downsample', 'down_sample_layer')

    return name


# def checkname(name):
#     if 'bn' in name:
#         if name.endswith('bias'):
#             name = name.replace('bias', 'beta')
#         if name.endswith('weight'):
#             name = name.replace('weight', 'gamma')
#         if name.endswith('running_mean'):
#             name = name.replace('running_mean', 'moving_mean')
#         if name.endswith('running_var'):
#             name = name.replace('running_var', 'moving_variance')
#     if 'bn' not in name and 'fc' not in name and '7x7' not in name:
#         if 'bias' in name:
#             name = name.replace('bias', 'conv.bias')
#         if name.endswith('running_mean'):
#             name = name.replace('running_mean', 'moving_mean')
#         if name.endswith('running_var'):
#             name = name.replace('running_var', 'moving_variance')
#         if name.endswith('bias'):
#             name = name.replace('bias', 'beta')
#         name = name.replace('downsample', 'down_sample_layer')
#     if 'down_sample_layer.0' in name:
#         #if name.endswith('weight'):
#             #name = name.replace('weight', 'gamma')
#         if name.endswith('conv.beta'):
#             name = name.replace('conv.beta', 'beta')
#     if 'down_sample_layer.1' in name:
#         if name.endswith('weight'):
#             name = name.replace('weight', 'gamma')
#         if name.endswith('conv.beta'):
#             name = name.replace('conv.beta', 'beta')
#     if '_bn' in name and '7x7' not in name:
#         name = name.replace('_bn', '.bn')
#     if 'conv1_7x7_s2' in name and 'bn' not in name:
#         name = 'conv1_7x7_s2.' + name
#     for item in modules:
#         if item in name:
#             name = item + '.' + name
#     name = name.replace('module.base_model.', '')
#     name = name.replace('net.', '')

#     return name


def convertpth2ckpt(arg):
    torch_params = torch.load(arg.model_path, map_location='cpu')
    new_params_list = []

    #for name in torch_params['state_dict']:
        #print(name)
    for name in torch_params['state_dict']:
        parameter = torch_params['state_dict'][name]
        if name == 'conv1_7x7_s2.weight':
            if arg.modality == 'Flow':
                kernel_size = parameter.size()
                new_kernel_size = kernel_size[:1] + (2 * arg.new_length,) + kernel_size[2:]
                parameter = parameter.data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_params_dict = {}
        #if 'fc' in name:
            #continue
        name = checkname(name)
        if parameter.numpy().dtype == 'float64':
            parameter = parameter.data.to(torch.float32)
        print(f'{name}')
        new_params_dict['name'] = name
        new_params_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(new_params_dict)
    for name in new_params_dict:
        print(name)
    save_checkpoint(new_params_list, 'tsm_RGB.ckpt'.format(str(arg.modality).lower()))


if __name__ == '__main__':
    args = parser.parse_args()
    args.model_path = '/home/sjyjxz/mindaspore/AMU1/caches/ImageNet/best_mocov3_resnet50_adapter_1shots.pt'
    args.modality = 'RGB'
    ckpt_file = 'tsm_RGB.ckpt'
    #print_weight_names_from_ckpt(ckpt_file)
    convertpth2ckpt(args)
    print("****" * 20)
