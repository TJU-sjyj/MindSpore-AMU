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
    # 使用MindSpore的TopK操作计算top-k的索引
    topk_op = ops.TopK(sorted=True)
    _, pred = topk_op(output, topk)
    pred = pred.transpose(0, 1)

    # 将target扩展为pred相同的形状以进行比较
    target_expanded = ops.expand_dims(target, 0)
    target_expanded = ops.broadcast_to(target_expanded, pred.shape)

    # 计算准确的预测
    correct = ops.equal(pred, target_expanded)

    # 计算并返回准确率
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
def run_ensemble_tip_adapter_F(args,
                            logger,
                            clip_test_features, 
                            aux_cache_keys, 
                            aux_cache_values, 
                            aux_test_features, 
                            test_labels, 
                            clip_weights, 
                            clip_model, 
                            aux_model, 
                            tfm_aux,
                            train_loader_F):
    
    
    # Enable the cached keys to be learnable
    
    #aux_adapter = Linear_Adapter(args.feat_dim, 1000, norm=True, init_type='cache_reduct',cache_weight=aux_cache_keys ,pre_phi='none', post_phi='none', order=args.order).cuda()
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
            #print("aux_image_features",aux_image_features.shape)
            aux_image_features = aux_image_features.detach().cpu().numpy()
            aux_image_features = mindspore.Tensor(aux_image_features, dtype=mstype.float32)
            #print("aux_image_features",aux_image_features.shape)
            aux_cache_logits = aux_adapter(aux_image_features)
            if type(aux_cache_logits) == list:
                sum_logits = 0
                for i in aux_cache_logits:
                    sum_logits += i
                aux_cache_logits = sum_logits
                        
            #print("clip_weights",clip_weights.dtype)
            clip_logits = 100. * clip_image_features @ clip_weights

            clip_logits_n = clip_logits.cpu().numpy()
            clip_logits = mindspore.Tensor(clip_logits_n, dtype=mstype.float32)

            tip_logits = clip_logits + aux_cache_logits * alpha
            #print("target",target.shape)
            #print("tip_logits",tip_logits.shape)
            target = target.detach().cpu().numpy()
            target = mindspore.Tensor(target, dtype=mindspore.int32)

            loss, gradients = grad_fn(tip_logits, target)

            acc = cls_acc(tip_logits, target)

            
        logger.info('Acc: {:.4f}'.format(acc))

        # Eval

        aux_logits = aux_adapter(aux_test_features)
        aux_logits = torch.from_numpy(aux_logits.asnumpy()).cuda()
        clip_logits = clip_test_features @ clip_weights
        #aux_logits = aux_logits.to('cuda:0')
        clip_weights = clip_weights.cuda()
        topk=1
        amu_logits = clip_logits + aux_logits * alpha
        
        pred = amu_logits.topk(topk, 1, True, True)[1].t()
        correct = pred.eq(test_labels.view(1, -1).expand_as(pred))
        acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        acc = 100 * acc / test_labels.shape[0]
        #acc, acc_aux = eval_result(args, aux_cache_values.dtype, aux_adapter, aux_test_features, clip_test_features, clip_weights, test_labels, alpha,uncent_power, split=1)
        logger.info("**** CaFo's test accuracy: {:.2f}. ****\n".format(acc))
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            save_path = args.cache_dir + f"/best_{args.aux_model_name}_{args.aux_backbone}_adapter_" + str(args.shots) + "shots.ckpt"
            save_checkpoint(aux_adapter, save_path)
    logger.info(f"**** After fine-tuning, CaFo's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")



def main():

    from parse_args import parse_args
    # Load config file
    parser = parse_args()
    args = parser.parse_args() # 这个parse_args()是类方法
    
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
    #imagenet = ImageNet(args.root_path, args.shots,preprocess)
    imagenet = ImageNet(args.root_path, args.shots)
    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=128, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=args.batch_size, num_workers=8, shuffle=True)


    # Textual features
    logger.info("Getting textual features as CLIP's classifier.")
    clip_weights = gpt_clip_classifier(imagenet.classnames, None, clip_model, imagenet.template)
    

    # Construct the cache model by few-shot training set
    logger.info("\nConstructing cache model by few-shot visual features and labels.")
    logger.info(f"\nConstructing AUX cache model ({args.aux_model_name}).")
    aux_cache_keys, aux_cache_values = build_cache_model(args, aux_model, train_loader_cache, tfm_norm=tfm_aux, model_name=args.aux_model_name, backbone_name=args.aux_backbone)


    # Pre-load test features
    logger.info("\nLoading visual features and labels from test set.")


    logger.info("\nLoading CLIP feature.")
    test_clip_features, test_labels, _= pre_load_features(args, "test", clip_model, test_loader, tfm_norm=tfm_clip, model_name='clip',backbone_name=args.clip_backbone)
    
    logger.info(f"\nLoading AUX feature ({args.aux_model_name}).")
    test_aux_features, test_labels, _= pre_load_features(args, "test", aux_model, test_loader, tfm_norm=tfm_aux, model_name=args.aux_model_name, backbone_name=args.aux_backbone)
    
    # 如果特征过大可以考虑挪到cpu上
    #test_clip_features = test_clip_features.cpu()
    #test_aux_features = test_aux_features.cpu()
    test_clip_features = test_clip_features.cuda()
    test_aux_features = test_aux_features.cuda()
    # 进行zero shot
    tmp =  test_clip_features / test_clip_features.norm(dim=-1, keepdim=True)
    l = 100. * tmp @ clip_weights
    # 测试准确率
    print(f"{l.argmax(dim=-1).eq(test_labels.cuda()).sum().item()}/ {len(test_labels)}")

    adapter = Linear_Adapter(args.feat_dim, 1000).cuda()
    adapter.weight = torch.load('/home/sjyjxz/mindaspore/AMU1/test/best_mocov3_resnet50_adapter_16shots.pt')
    aux_logits = adapter(test_aux_features)
    clip_logits = test_clip_features @ clip_weights
    tip_logits = clip_logits + aux_logits * 0.65
    acc = test_acc(tip_logits,test_labels)
    print(f"Acc: {acc}%")
    zs_acc = test_acc(clip_logits,test_labels)
    print(f"Acc: {zs_acc}%")
    adapter = Linear_Adapter(args.feat_dim, 1000).cuda()
    adapter.weight = torch.load('/home/sjyjxz/mindaspore/AMU1/caches/ImageNet/best_mocov3_resnet50_adapter_16shots.pt')
    aux_logits = adapter(test_aux_features)
    clip_logits = test_clip_features @ clip_weights
    tip_logits = clip_logits + aux_logits * 0.65
    acc = test_acc(tip_logits,test_labels)
    print(f"Acc: {acc}%")
    zs_acc = test_acc(clip_logits,test_labels)
    print(f"Acc: {zs_acc}%")
    adapter = Linear_Adapter(args.feat_dim, 1000).cuda()
    adapter.weight = torch.load('/home/sjyjxz/mindaspore/AMU1/caches/ImageNet/best_mocov3_resnet50_adapter_8shots.pt')
    aux_logits = adapter(test_aux_features)
    clip_logits = test_clip_features @ clip_weights
    tip_logits = clip_logits + aux_logits * 0.65
    acc = test_acc(tip_logits,test_labels)
    print(f"Acc: {acc}%")
    zs_acc = test_acc(clip_logits,test_labels)
    print(f"Acc: {zs_acc}%")





    time.sleep(655)
    
    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
   
    run_ensemble_tip_adapter_F(args,
                            logger,
                            test_clip_features, 
                            aux_cache_keys, 
                            aux_cache_values, 
                            test_aux_features, 
                            test_labels,
                            clip_weights, 
                            clip_model, 
                            aux_model,
                            tfm_aux,
                            train_loader_F)

if __name__ == '__main__':
    main()

