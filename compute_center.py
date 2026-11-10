import argparse
import numpy as np
import torch
import random
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model.res12 import Res12
from model.swin_transformer import swin_tiny
# 导入双骨干融合网络
from model.Swin_Res12 import Swin_Res12

from utils.utils import transform_val, transform_val_cifar, cluster
from utils.utils import transform_val_224_cifar, transform_val_224


def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================== 模型权重加载 (支持 swin_res12) ==================
    if args.backbone == 'resnet':
        model = Res12(avg_pool=True, drop_block='ImageNet' in args.dataset).to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(f'./checkpoints/ResNet-{args.dataset}.pth', weights_only=False)['params']
        checkpoint = {k[8:]: v for k, v in checkpoint.items()}
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        model.load_state_dict(checkpoint, strict=False)

    elif args.backbone == 'swin':
        model = swin_tiny().to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(f'./checkpoints/Swin-Tiny-{args.dataset}.pth', weights_only=False)['params']
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        model.load_state_dict(checkpoint, strict=False)

    elif args.backbone == 'swin_res12':
        model = Swin_Res12(dataset=args.dataset).to(device)

        ckpt_res = torch.load(f'./checkpoints/ResNet-{args.dataset}.pth', weights_only=False)['params']
        ckpt_res = {k[8:]: v for k, v in ckpt_res.items() if k[8:] in model.cnn.state_dict()}
        model.cnn.load_state_dict(ckpt_res, strict=False)

        ckpt_swin = torch.load(f'./checkpoints/Swin-Tiny-{args.dataset}.pth', weights_only=False)['params']
        ckpt_swin = {k: v for k, v in ckpt_swin.items() if k in model.swin.state_dict()}
        model.swin.load_state_dict(ckpt_swin, strict=False)

    model.eval()

    # ================== 数据集加载 ==================
    batch_size = 128
    shuffle = True

    if args.dataset == 'MiniImageNet':
        trainset = ImageFolder('/home/aa/cpx/FewShot/datasets/MiniImageNet/train',
                               transform=transform_val if args.backbone == 'resnet' else transform_val_224)
    elif args.dataset == 'FC100':
        trainset = ImageFolder('/home/aa/cpx/FewShot/datasets/FC100/FC100/train',
                               transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)
    elif args.dataset == 'CIFAR-FS':
        trainset = ImageFolder('/home/aa/cpx/FewShot/datasets/CIFAR-FS/cifar-fs/train',
                               transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)
    elif args.dataset == 'TieredImageNet':
        trainset = ImageFolder('/home/aa/cpx/FewShot/datasets/tiered_imagenet/train',
                               transform=transform_val if args.backbone == 'resnet' else transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    idx_to_class = {v: k for k, v in trainset.class_to_idx.items()}

    data = {}
    for x, labels in tqdm(train_loader):
        labels = [idx_to_class[l.item()] for l in labels]
        with torch.no_grad():
            # 统一提取特征并展平
            x = model(x.to(device)).view(x.size(0), -1)
        for i, l in enumerate(labels):
            if l in data:
                data[l].append(x[i].detach().cpu().numpy())
            else:
                data[l] = [x[i].detach().cpu().numpy()]

    print('Finished train feature extraction')

    center_mean = {}
    for k, v in data.items():
        center_mean[k] = np.array(v).mean(0)

    if args.dataset == 'TieredImageNet':
        data = {k: v[:700] for k, v in data.items()}
        center_cluster = cluster(data, len(data), 700)
    else:
        center_cluster = cluster(data, len(data), 600)

    # 保存计算好的中心
    save_path = f'center_{args.dataset}_{args.backbone}.pth'
    torch.save({
        'mean': center_mean,
        'cluster': center_cluster,
        'center': center_mean
    }, save_path)
    print(f"Center saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    parser.add_argument('--center', default='mean')
    # 添加 swin_res12 选项
    parser.add_argument('--backbone', default='resnet', choices=['resnet', 'swin', 'swin_res12'])
    args = parser.parse_args()
    print(vars(args))
    main()