from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
import torch.nn as nn

import clip

from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor, RandomResizedCrop, RandomHorizontalFlip

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")



def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

tfm_train_base = Compose([
            RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=BICUBIC),
            RandomHorizontalFlip(p=0.5),
            ToTensor()
            ]
        )

tfm_test_base = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
    ])

# transform
tfm_clip = Compose([
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

tfm_normal = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]    
)

def cls_acc(output, target, topk=1):

    output = torch.from_numpy(output.asnumpy())
    target = torch.from_numpy(target.asnumpy())

    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def gpt_clip_classifier(classnames, gpt_prompts, clip_model, template):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            
            #texts = []
            #for t in gpt_prompts[classname]:
            #    texts.append(t)
            
            texts = [t.format(classname) for t in template]
            
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights





def build_cache_model(args, model, train_loader_cache, tfm_norm,model_name, backbone_name):
    
    if args.load_cache == False:    
        cache_keys = []
        cache_values = []
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(args.augment_epoch):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, args.augment_epoch))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    if hasattr(model, 'encode_image') and callable(getattr(model, 'encode_image')):
                        image_features = model.encode_image(tfm_norm(images)) # for clip model
                    else:
                        image_features = model(tfm_norm(images))
                    if len(image_features.shape) == 4:
                        image_features = avgpool(image_features).squeeze()
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, args.cache_dir + f'/{model_name}_{backbone_name}_keys_' + str(args.shots) + "shots.pt")
        torch.save(cache_values, args.cache_dir + f'/{model_name}_{backbone_name}_values_' + str(args.shots) + "shots.pt")

    else:
        cache_keys = torch.load(args.cache_dir + f'/{model_name}_{backbone_name}_keys_' + str(args.shots) + "shots.pt")
        cache_values = torch.load(args.cache_dir + f'/{model_name}_{backbone_name}_values_' + str(args.shots) + "shots.pt")
    return cache_keys, cache_values


def build_clip_dalle_cache_model(args, clip_model, train_loader_cache):
    
    if args.load_cache == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(args.augment_epoch):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, args.augment_epoch))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, args.cache_dir + '/clip_dalle_keys_' + str(cfg['dalle_shots']) + "shots.pt")
        torch.save(cache_values, args.cache_dir + '/clip_dalle_values_' + str(cfg['dalle_shots']) + "shots.pt")

    else:
        cache_keys = torch.load(args.cache_dir + '/clip_dalle_keys_' + str(cfg['dalle_shots']) + "shots.pt")
        cache_values = torch.load(args.cache_dir + '/clip_dalle_values_' + str(cfg['dalle_shots']) + "shots.pt")

    return cache_keys, cache_values

def build_dino_dalle_cache_model(args, dino_model, train_loader_cache):
    
    if args.load_cache == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(args.augment_epoch):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, args.augment_epoch))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = dino_model(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, args.cache_dir + '/dino_dalle_keys_' + str(cfg['dalle_shots']) + "shots.pt")
        torch.save(cache_values, args.cache_dir + '/dino_dalle_values_' + str(cfg['dalle_shots']) + "shots.pt")

    else:
        cache_keys = torch.load(args.cache_dir + '/dino_dalle_keys_' +  "shots.pt")
        cache_values = torch.load(args.cache_dir + '/dino_dalle_values_' + "shots.pt")

    return cache_keys, cache_values



def pre_load_features(args, split, model, loader, tfm_norm, model_name, backbone_name):
    '''
    不再对test feat进行归一化
    '''
    if args.load_pre_feat == False:
        features, labels = [], []
        middle_features = []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                if hasattr(model, 'encode_image') and callable(getattr(model, 'encode_image')):
                    image_features = model.encode_image(tfm_norm(images)) # for clip model
                else:
                    image_features = model(tfm_norm(images))
                '''
                # 如果有feat_middle属性，则取出
                if hasattr(model, 'feat_middle'):
                    middle_features.append(model.feat_middle.cpu())
                '''
                features.append(image_features.cpu()) # 对于某些爆显存的，只能这么干了。不过到时候记得在命令行把这个玩意移动到cuda上，不然后边报错
                #features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)
        features = features.cuda() # 显式移动到cuda上
        torch.save(features, args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_f.pt")
        torch.save(labels, args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_l.pt")
        if len(middle_features) > 0:
            middle_features = torch.cat(middle_features)
            torch.save(middle_features, args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_middle_f.pt")

    else:
        features = torch.load(args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_f.pt")
        labels = torch.load(args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_l.pt")
        '''
        if os.path.exists(args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_middle_f.pt"):
            middle_features = torch.load(args.cache_dir + "/" + split + f"_{model_name}_{backbone_name}_middle_f.pt")
        else:
            middle_features = None
        '''
        middle_features = None

    return features, labels, middle_features


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

def search_no_clip_hp(cfg, cache_keys, cache_values, features, labels, adapter=None):
    
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features).to(torch.float16)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                # clip_logits = 100. * features @ clip_weights
                # tip_logits = clip_logits + cache_logits * alpha
                tip_logits = cache_logits
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def search_ensemble_hp(cfg, 
                    clip_cache_keys, 
                    clip_cache_values, 
                    clip_features, 
                    dino_cache_keys, 
                    dino_cache_values, 
                    dino_features, 
                    labels, 
                    clip_weights, 
                    clip_adapter=None, 
                    dino_adapter=None):
    
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if clip_adapter:
                    clip_affinity = clip_adapter(clip_features)
                    dino_affinity = dino_adapter(dino_features).to(dino_cache_values)
                else:
                    clip_affinity = clip_features @ clip_cache_keys
                    dino_affinity = (dino_features @ dino_cache_keys).to(dino_cache_values)

                clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
                dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
                clip_logits = 100. * clip_features @ clip_weights
                cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
        with open("best.txt","w") as f:
            f.write("After searching, the best accuarcy: {:.2f}.\n".format(best_acc))
    return best_beta, best_alpha


# clip zero_shot as baseline
def logits_fuse(zero_logtis, logits, normalize='mean'):
    # normalize logits
    softmax_fun = nn.Softmax(dim=1)
    if normalize == 'softmax':
        zero_logtis = softmax_fun(zero_logtis)
    elif normalize =='linear':
        zero_logtis /= torch.norm(zero_logtis, p=2, dim=1, keepdim=True)
    elif normalize == 'mean':
        logits_std = torch.std(zero_logtis, dim=1, keepdim=True)
        logits_mean = torch.mean(zero_logtis, dim=1, keepdim=True)
        zero_logtis = (zero_logtis - logits_mean) / logits_std
    else:
        raise("error normalize!")
    similarity_matrix = []
    normalize_logits = []
    for logit in logits:
        if normalize == 'softmax':
            current_normalize_logits = softmax_fun(logit)
        elif normalize =='linear':
            current_normalize_logits = logit / torch.norm(logit, p=2, dim=1, keepdim=True)
        elif normalize == 'mean':
            logits_std = torch.std(logit, dim=1, keepdim=True)
            logits_mean = torch.mean(logit, dim=1, keepdim=True)
            current_normalize_logits = (logit - logits_mean) / logits_std
        else:
            raise("error normalize!")
        current_similarity = current_normalize_logits * zero_logtis
        current_similarity = torch.sum(current_similarity, dim=1, keepdim=True)
        similarity_matrix.append(current_similarity)
        normalize_logits.append(current_normalize_logits)
    similarity_matrix = torch.stack(similarity_matrix, dim=-2)
    similarity_matrix = softmax_fun(similarity_matrix)
    normalize_logits = torch.stack(normalize_logits, dim=-2)
    result_logits = torch.sum(normalize_logits * similarity_matrix, dim=1)

    return result_logits
def logits_fuse_s(zero_logtis, logits, normalize='mean'):
    # normalize logits
    softmax_fun = nn.Softmax(dim=1)
    if normalize == 'softmax':
        zero_logtis = softmax_fun(zero_logtis)
    elif normalize =='linear':
        zero_logtis /= torch.norm(zero_logtis, p=2, dim=1, keepdim=True)
    elif normalize == 'mean':
        logits_std = torch.std(zero_logtis, dim=1, keepdim=True)
        logits_mean = torch.mean(zero_logtis, dim=1, keepdim=True)
        zero_logtis = (zero_logtis - logits_mean) / logits_std
    else:
        raise("error normalize!")
    similarity_matrix = []
    normalize_logits = []
    for logit in logits:
        if normalize == 'softmax':
            current_normalize_logits = softmax_fun(logit)
        elif normalize =='linear':
            current_normalize_logits = logit / torch.norm(logit, p=2, dim=1, keepdim=True)
        elif normalize == 'mean':
            logits_std = torch.std(logit, dim=1, keepdim=True)
            logits_mean = torch.mean(logit, dim=1, keepdim=True)
            current_normalize_logits = (logit - logits_mean) / logits_std
        else:
            raise("error normalize!")
        current_similarity = current_normalize_logits * zero_logtis
        current_similarity = torch.sum(current_similarity, dim=1, keepdim=True)
        similarity_matrix.append(current_similarity)
        normalize_logits.append(current_normalize_logits)
    similarity_matrix = torch.stack(similarity_matrix, dim=-2)
    similarity_matrix = softmax_fun(similarity_matrix)
    count = 0
    for i in similarity_matrix:
        if i[0]>0.4 and i[0]<0.6:
            count += 1
    normalize_logits = torch.stack(normalize_logits, dim=-2)
    result_logits = torch.sum(normalize_logits * similarity_matrix, dim=1)

    return result_logits, count

import logging
import datetime

def config_logging(args):
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M')
    now = datetime.datetime.now().strftime("%m-%d-%H_%M")
    # 使用FileHandler输出到文件
    fh = logging.FileHandler(f'result/{args.exp_name}_{now}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger 
