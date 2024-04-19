import os
import random
import argparse
import yaml
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models as torchvision_models
import mindspore.nn as mnn
import mindspore.numpy as mnp
import mindspore
from mindspore.common import dtype as mstype
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore import ops,save_checkpoint,ParameterTuple,Tensor
from datasets.imagenet_lzy import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from backbones.load_model import load_model
from mindspore import context
from backbones.heads import *
def cls_macc(output, target, topk=1):

    topk_op = ops.TopK(sorted=True)
    _, pred = topk_op(output, topk)
    pred = pred.transpose(0, 1)
    target_expanded = ops.expand_dims(target, 0)
    target_expanded = ops.broadcast_to(target_expanded, pred.shape)
    correct = ops.equal(pred, target_expanded)
    acc = ops.cast(correct[:topk], float32).sum() / Tensor(target.shape[0], dtype=float32)
    
    return 100 * acc.asnumpy()
    return acc * 100

class MyLoss(mnn.Cell):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss_fn = mnn.SoftmaxCrossEntropyWithLogits(sparse=True)

    def construct(self, logits, labels):
        return self.loss_fn(logits, labels)

def test_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc
def run_ensemble_amu_adapter_F(args,
                            logger,
                            clip_test_features, 
                            aux_test_features, 
                            test_labels, 
                            clip_weights, 
                            clip_model, 
                            aux_model, 
                            tfm_aux,
                            train_loader_F):
    

    aux_adapter = mindspore.nn.Dense(2048, out_channels=1000)
    uncent_power = args.uncent_power #0.3
    uncent_type = args.uncent_type #'none'
    loss_cell = MyLoss()
    optimizer = mnn.Adam(params=aux_adapter.trainable_params(), learning_rate=0.001)
    grad_fn = mindspore.ops.value_and_grad(loss_cell, None, optimizer.parameters)
    aux_test_features_n = aux_test_features.cpu().numpy()
    aux_test_features = mindspore.Tensor(aux_test_features_n, dtype=mstype.float32)

    beta, alpha = args.init_beta, args.init_alpha
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(1, args.train_epoch + 1):
        # Train
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        correct_samples, all_samples = 0, 0
        loss_list = []
        loss_aux_list = []
        loss_merge_list = [] 
        logger.info('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))

        # origin image
        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad(): 
                clip_image_features = clip_model.encode_image(tfm_clip(images))
                clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)                
                if hasattr(aux_model, 'encode_image') and callable(getattr(aux_model, 'encode_image')):
                    aux_image_features= aux_model.encode_image(tfm_aux(images)) # for clip model
                else:
                    aux_image_features = aux_model(tfm_aux(images))
            aux_image_features = aux_image_features.detach().cpu().numpy()
            aux_image_features = mindspore.Tensor(aux_image_features, dtype=mstype.float32)
            aux_cache_logits = aux_adapter(aux_image_features)
            if type(aux_cache_logits) == list:
                sum_logits = 0
                for i in aux_cache_logits:
                    sum_logits += i
                aux_cache_logits = sum_logits
                        
            clip_logits = 100. * clip_image_features @ clip_weights

            clip_logits_n = clip_logits.cpu().numpy()
            clip_logits = mindspore.Tensor(clip_logits_n, dtype=mstype.float32)

            amu_logits = clip_logits + aux_cache_logits * alpha
            target = target.detach().cpu().numpy()
            target = mindspore.Tensor(target, dtype=mindspore.int32)

            loss, gradients = grad_fn(amu_logits, target)
            

        # Eval

        aux_logits = aux_adapter(aux_test_features)
        aux_logits = torch.from_numpy(aux_logits.asnumpy()).cuda()
        clip_logits = clip_test_features @ clip_weights

        clip_weights = clip_weights.cuda()
        topk=1
        amu_logits = clip_logits + aux_logits * alpha
        
        pred = amu_logits.topk(topk, 1, True, True)[1].t()
        correct = pred.eq(test_labels.view(1, -1).expand_as(pred))
        acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        acc = 100 * acc / test_labels.shape[0]
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            save_path = args.cache_dir + f"/best_{args.aux_model_name}_{args.aux_backbone}_adapter_" + str(args.shots) + "shots.ckpt"
            save_checkpoint(aux_adapter, save_path)
    logger.info(f"Best Test Accuracy: {best_acc:.2f}, At Epoch: {best_epoch}.")



def main():

    from parse_args import parse_args
    # Load config file
    parser = parse_args()
    args = parser.parse_args() # This parse_args() is the class method
    
    cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir

    logger = config_logging(args)
    logger.info("\nRunning configs.")
    args_dict = vars(args)
    message = '\n'.join([f'{k:<20}: {v}' for k, v in args_dict.items()])
    logger.info(message)

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
    
    logger.info("Preparing ImageNet dataset.")
    imagenet = ImageNet(args.root_path, args.shots)
    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=128, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader = torch.utils.data.DataLoader(imagenet.train, batch_size=args.batch_size, num_workers=8, shuffle=True)


    # Textual features
    logger.info("Getting textual features as CLIP's classifier.")
    clip_weights = gpt_clip_classifier(imagenet.classnames, None, clip_model, imagenet.template)
    

    # Construct the cache model by few-shot training set
    logger.info("\nConstructing cache model by few-shot visual features and labels.")
    logger.info(f"\nConstructing AUX cache model ({args.aux_model_name}).")

    # Pre-load test features
    logger.info("\nLoading visual features and labels from test set.")


    logger.info("\nLoading CLIP feature.")
    test_clip_features, test_labels, _= pre_load_features(args, "test", clip_model, test_loader, tfm_norm=tfm_clip, model_name='clip',backbone_name=args.clip_backbone)
    
    logger.info(f"\nLoading AUX feature ({args.aux_model_name}).")
    test_aux_features, test_labels, _= pre_load_features(args, "test", aux_model, test_loader, tfm_norm=tfm_aux, model_name=args.aux_model_name, backbone_name=args.aux_backbone)
    
    #If the features are too large consider moving them to the cpu
    #test_clip_features = test_clip_features.cpu()
    #test_aux_features = test_aux_features.cpu()
    test_clip_features = test_clip_features.cuda()
    test_aux_features = test_aux_features.cuda()




   
    run_ensemble_amu_adapter_F(args,
                            logger,
                            test_clip_features, 
                            test_aux_features, 
                            test_labels,
                            clip_weights, 
                            clip_model, 
                            aux_model,
                            tfm_aux,
                            train_loader)

if __name__ == '__main__':
    main()

