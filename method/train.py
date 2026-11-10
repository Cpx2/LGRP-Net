import os
import sys
import random
import numpy as np
import torch


# 固定随机种子，保证训练结果完全一致
def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import argparse
import os.path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from method.SemAlign import SemAlign

from data.samplers import CategoriesSampler
# from data.tiered_imagenet import tieredImageNet
from logger import loggers
from model.res12 import Res12
from model.swin_transformer import swin_tiny
from utils.utils import Cosine_classifier, count_95acc, transform_val_cifar, transform_train_cifar, \
    transform_train, count_kacc, transform_val
from utils.utils import transform_val_224_cifar, transform_train_224_cifar, transform_train_224, transform_val_224

from sklearn.cluster import KMeans
from method.QSWPR import updateproto_


# ============================================================


# ====================== 调试工具函数 ======================
def print_check_info(title, content):
    """打印检查信息，带颜色标识"""
    print(f"\n\033[32m========== {title} ==========\033[0m")  # 绿色标题
    print(content)


def print_warning(info):
    """打印警告信息"""
    print(f"\033[33m[WARNING] {info}\033[0m")  # 黄色警告


def print_error(info):
    """打印错误信息"""
    print(f"\033[31m[ERROR] {info}\033[0m")  # 红色错误


if __name__ == '__main__':
    # 全局生效随机种子
    set_seed(13)

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--test-batch', type=int, default=600)
    parser.add_argument('--center', default='mean')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--feat-size', type=int, default=640)
    parser.add_argument('--semantic-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--mode', type=str, default='clip')
    parser.add_argument('--text_type', type=str, default='gpt')
    parser.add_argument('--dataset', type=str, default='TieredImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    # [修改] 增加 swin_res12 选项
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'swin', 'swin_res12'])
    args = parser.parse_args()

    # ====================== 1. 基础参数初始化 ======================
    if args.backbone == 'resnet':
        args.model_path = './checkpoints/ResNet-{}.pth'.format(args.dataset)
    elif args.backbone == 'swin':
        args.model_path = './checkpoints/Swin-Tiny-{}.pth'.format(args.dataset)

    args.work_dir = '{}_{}_{}_{}_{}_{}'.format(args.backbone, args.dataset, args.mode, args.text_type, args.center,
                                               args.shot)

    if args.dataset == 'TieredImageNet':
        args.num_workers = 0

    if os.path.exists(args.work_dir) is False:
        os.mkdir(args.work_dir)

    log = loggers(os.path.join(args.work_dir, 'train'))
    log.info(vars(args))

    writer = SummaryWriter(os.path.join(args.work_dir, 'train_semantic'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_check_info("设备信息", f"使用设备: {device}")

    # ====================== 2. 数据集加载及类别检查 ======================
    if args.dataset == 'MiniImageNet':
        args.val = '/home/aa/cpx/FewShot/datasets/MiniImageNet/val'
        args.train = '/home/aa/cpx/FewShot/datasets/MiniImageNet/train'
        train_dataset = ImageFolder(args.train,
                                    transform=transform_train if args.backbone == 'resnet' else transform_train_224)
        val_dataset = ImageFolder(args.val, transform=transform_val if args.backbone == 'resnet' else transform_val_224)

    elif args.dataset == 'FC100':
        args.val = '/home/aa/cpx/FewShot/datasets/FC100/FC100/val'
        args.train = '/home/aa/cpx/FewShot/datasets/FC100/FC100/train'
        train_dataset = ImageFolder(args.train,
                                    transform=transform_train_cifar if args.backbone == 'resnet' else transform_train_224_cifar)
        val_dataset = ImageFolder(args.val,
                                  transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)

    elif args.dataset == 'CIFAR-FS':
        args.val = '/home/aa/cpx/FewShot/datasets/CIFAR-FS/cifar-fs/val'
        args.train = '/home/aa/cpx/FewShot/datasets/CIFAR-FS/cifar-fs/train'
        train_dataset = ImageFolder(args.train,
                                    transform=transform_train_cifar if args.backbone == 'resnet' else transform_train_224_cifar)
        val_dataset = ImageFolder(args.val,
                                  transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)

    elif args.dataset == 'TieredImageNet':
        args.val = '/home/aa/cpx/FewShot/datasets/tiered_imagenet/val'
        args.train = '/home/aa/cpx/FewShot/datasets/tiered_imagenet/train'
        train_dataset = ImageFolder(args.train,
                                    transform=transform_train if args.backbone == 'resnet' else transform_train_224)
        val_dataset = ImageFolder(args.val, transform=transform_val if args.backbone == 'resnet' else transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')

    # 提取数据集类别信息
    val_idx_to_class = val_dataset.class_to_idx
    val_idx_to_class_rev = {v: k for k, v in val_idx_to_class.items()}  # 索引→类别名
    idx_to_class = train_dataset.class_to_idx
    idx_to_class_rev = {v: k for k, v in idx_to_class.items()}  # 索引→类别名

    # 只保留训练集/验证集的独立统计（不合并）
    train_class_names = set(idx_to_class.keys())
    val_class_names = set(val_idx_to_class.keys())

    print_check_info("数据集类别信息", f"""
训练集类别数: {len(train_class_names)} | 前5个类别: {list(train_class_names)[:5]}
验证集类别数: {len(val_class_names)} | 前5个类别: {list(val_class_names)[:5]}
注：类中心仅包含训练集类别，验证集无类中心属于正常情况
    """)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    val_sampler = CategoriesSampler(val_dataset.targets, args.test_batch, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)

    # ====================== 3. 类中心（proto_center）检查 ======================
    center_file_path = f'center_{args.dataset}_{args.backbone}.pth'
    print_check_info("类中心文件检查", f"类中心文件路径: {center_file_path}")

    if not os.path.exists(center_file_path):
        print_error(f"类中心文件不存在: {center_file_path}")
        raise FileNotFoundError(f"类中心文件缺失: {center_file_path}")

    # 加载类中心并检查
    proto_center = torch.load(center_file_path, weights_only=False)[args.center]
    center_class_names = set(proto_center.keys())

    # 仅检查训练集类别匹配
    missing_in_center = train_class_names - center_class_names
    extra_in_center = center_class_names - train_class_names

    print_check_info("类中心类别匹配（仅训练集）", f"""
类中心包含类别数: {len(center_class_names)} | 前5个类别: {list(center_class_names)[:5]}
训练集有但类中心缺失的类别数: {len(missing_in_center)}
{'- 缺失类别: ' + str(list(missing_in_center)[:10]) if missing_in_center else '- 无缺失类别'}
类中心有但训练集没有的类别数: {len(extra_in_center)}
{'- 多余类别: ' + str(list(extra_in_center)[:10]) if extra_in_center else '- 无多余类别'}
训练集类别匹配率: {len(train_class_names & center_class_names) / len(train_class_names) * 100:.2f}%
    """)

    if len(missing_in_center) > 0:
        print_warning(f"有 {len(missing_in_center)} 个训练集类别在类中心中缺失，会导致训练报错！")
    else:
        print_check_info("类中心检查结果", "✅ 类中心包含所有训练集类别，匹配正常")

    # ====================== 4. 模型权重加载及参数检查 ======================
    if args.backbone != 'swin_res12':
        print_check_info("模型权重检查", f"模型权重路径: {args.model_path}")
        if not os.path.exists(args.model_path):
            print_error(f"模型权重文件不存在: {args.model_path}")
            raise FileNotFoundError(f"模型权重缺失: {args.model_path}")

    # 加载模型
    if args.backbone == 'resnet':
        model = Res12(avg_pool=True, drop_block='ImageNet' in args.dataset).to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.model_path)['params']
        checkpoint = {k[8:]: v for k, v in checkpoint.items()}
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        model.load_state_dict(checkpoint, strict=False)
        feat_size = 640

    elif args.backbone == 'swin':
        model = swin_tiny().to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.model_path, weights_only=False)['params']
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        model.load_state_dict(checkpoint, strict=False)
        feat_size = 768

    # [新增] 融合骨干加载
    elif args.backbone == 'swin_res12':
        from model.Swin_Res12 import Swin_Res12

        model = Swin_Res12(dataset=args.dataset).to(device)

        ckpt_res = torch.load(f'./checkpoints/ResNet-{args.dataset}.pth', weights_only=False)['params']
        ckpt_res = {k[8:]: v for k, v in ckpt_res.items() if k[8:] in model.cnn.state_dict()}
        model.cnn.load_state_dict(ckpt_res, strict=False)

        ckpt_swin = torch.load(f'./checkpoints/Swin-Tiny-{args.dataset}.pth', weights_only=False)['params']
        ckpt_swin = {k: v for k, v in ckpt_swin.items() if k in model.swin.state_dict()}
        model.swin.load_state_dict(ckpt_swin, strict=False)

        feat_size = 1408  # 特征维度更新
        checkpoint = {}  # 避免下方len(checkpoint)报错

    print(f"\n加载的参数数量: {len(checkpoint)}")  # 保留原打印
    model.eval()

    # ====================== 5. 语义文件（semantic）检查 ======================
    if 'ImageNet' in args.dataset:
        semantic_file_path = f'./semantic/imagenet_semantic_{args.mode}_{args.text_type}.pth'
    else:
        semantic_file_path = f'./semantic/cifar100_semantic_{args.mode}_{args.text_type}.pth'

    print_check_info("语义文件检查", f"语义文件路径: {semantic_file_path}")

    if not os.path.exists(semantic_file_path):
        print_error(f"语义文件不存在: {semantic_file_path}")
        raise FileNotFoundError(f"语义文件缺失: {semantic_file_path}")

    semantic = torch.load(semantic_file_path, weights_only=False)['semantic_feature']
    semantic = {k: v.float() for k, v in semantic.items()}
    semantic_class_names = set(semantic.keys())

    train_missing_in_semantic = train_class_names - semantic_class_names
    val_missing_in_semantic = val_class_names - semantic_class_names
    extra_in_semantic = semantic_class_names - (train_class_names.union(val_class_names))

    print_check_info("语义文件类别匹配", f"""
语义文件包含类别数: {len(semantic_class_names)} | 前5个类别: {list(semantic_class_names)[:5]}
训练集有但语义文件缺失的类别数: {len(train_missing_in_semantic)}
{'- 训练集缺失: ' + str(list(train_missing_in_semantic)[:10]) if train_missing_in_semantic else '- 训练集无缺失'}
验证集有但语义文件缺失的类别数: {len(val_missing_in_semantic)}
{'- 验证集缺失: ' + str(list(val_missing_in_semantic)[:10]) if val_missing_in_semantic else '- 验证集无缺失'}
语义文件有但数据集没有的类别数: {len(extra_in_semantic)}
{'- 多余类别: ' + str(list(extra_in_semantic)[:10]) if extra_in_semantic else '- 无多余类别'}
训练集语义匹配率: {len(train_class_names & semantic_class_names) / len(train_class_names) * 100:.2f}%
    """)

    if len(train_missing_in_semantic) > 0:
        print_warning(f"有 {len(train_missing_in_semantic)} 个训练集类别在语义文件中缺失，会导致训练报错！")
    else:
        print_check_info("语义文件检查结果", "✅ 语义文件包含所有训练集类别，匹配正常")

    # ====================== 6. 后续训练逻辑 ======================
    H = SemAlign(feat_size, args.semantic_size, h_size=4096, drop=args.drop).to(device)
    optimizer = torch.optim.Adam(H.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    gap_acc = -1
    max_acc1 = 0.0
    for epoch in range(args.max_epoch):
        for step, [data, labels] in enumerate(tqdm(train_loader)):
            try:
                proto = torch.tensor(np.array([proto_center[idx_to_class_rev[l.item()]] for l in labels])).to(device)
                text_feature = torch.stack([semantic[idx_to_class_rev[l.item()]] for l in labels]).to(device)
            except KeyError as e:
                print_error(
                    f"类别索引匹配错误: {e} | 出错标签索引: {l.item()} | 对应类别名: {idx_to_class_rev.get(l.item(), '未知')}")
                raise e

            with torch.no_grad():
                img_feature = model(data.to(device))

            optimizer.zero_grad()
            H.zero_grad()
            H.train()
            fusion = H(text_feature, img_feature)
            recon_loss = F.l1_loss(fusion, proto)

            g_loss = recon_loss
            g_loss.backward()
            optimizer.step()

        log.info(
            "[Epoch %d/%d] [recon loss: %f] "
            % (epoch, args.max_epoch, recon_loss.item(),)
        )

        lr_scheduler.step()

        # val
        ks = np.arange(0, 101) * 0.01
        label = torch.arange(args.test_way).repeat(args.query).type(torch.cuda.LongTensor)
        if epoch % 10 == 0 or epoch > 0:
            P_acc = {}
            O_acc = []
            G_acc = []
            with torch.no_grad():
                for data, labels in tqdm(val_loader):
                    data = data.to(device)
                    data = model(data).view(data.size(0), -1)
                    n_support = args.shot * args.test_way
                    support, query = data[:n_support], data[n_support:]

                    ys = torch.arange(args.test_way).repeat(args.shot).to(device)
                    proto = updateproto_(support, query, ys, args.test_way, temp=15.0)
                    # =================================================================================
                    try:
                        s = torch.stack([semantic[val_idx_to_class_rev[l.item()]] for l in labels[:n_support]]).to(
                            device)
                    except KeyError as e:
                        print_error(
                            f"验证集类别索引匹配错误: {e} | 出错标签索引: {l.item()} | 对应类别名: {val_idx_to_class_rev.get(l.item(), '未知')}")
                        raise e

                    gen_proto = H(s, support)
                    gen_proto = gen_proto.reshape(args.shot, args.test_way, -1).mean(0)

                    _, predict0 = Cosine_classifier(proto, query)
                    _, predict1 = Cosine_classifier(gen_proto, query)
                    O_acc.append(((predict0 == label).sum() / len(label)).item())
                    G_acc.append(((predict1 == label).sum() / len(label)).item())

                    for f in ks:
                        if str(f) in P_acc:
                            P_acc[str(f)].append(count_kacc(proto, gen_proto, query, f, args))
                        else:
                            P_acc[str(f)] = []
                            P_acc[str(f)].append(count_kacc(proto, gen_proto, query, f, args))

                O_acc, O_95 = count_95acc(np.array(O_acc))
                G_acc, G_95 = count_95acc(np.array(G_acc))

                max_acc = {
                    'k': 0,
                    'acc': 0,
                    'acc95': 0,
                }
                for k, v in P_acc.items():
                    P_acc[k] = count_95acc(np.array(v))
                    if P_acc[k][0] > max_acc['acc']:
                        max_acc['acc'] = P_acc[k][0]
                        max_acc['acc95'] = P_acc[k][1]
                        max_acc['k'] = k
                cur_gap = max_acc['acc'] - O_acc
                if cur_gap > gap_acc:
                    gap_acc = cur_gap
                if max_acc['acc'] > max_acc1:
                    max_acc1 = max_acc['acc']
                    torch.save({
                        'G': H,
                        'epoch': epoch,
                        'k': max_acc['k'],
                        'acc': max_acc1
                    }, os.path.join(args.work_dir, 'epoch_best.pth'.format(epoch)))
                    log.info('best epoch: %d' % epoch)
                    print('save', epoch)

            writer.add_scalars("add_scalars/acc", {'origin': O_acc,
                                                   'complete': max_acc['acc']}, epoch)
            log.info(
                'epoch: %d |origin acc: %.2f+%.2f%% |complete acc: %.2f+%.2f%% |gap: %.2f/%.2f |k: %s' % (
                    epoch, O_acc * 100, O_95 * 100, max_acc['acc'] * 100, max_acc['acc95'] * 100,
                    gap_acc * 100, cur_gap * 100, max_acc['k']))
            log.info('ACC |proto acc: %.2f+%.2f%% |gen acc: %.2f+%.2f%% |Max: %.2f' % (
                O_acc * 100, O_95 * 100, G_acc * 100, G_95 * 100, 100 * max_acc1))

    writer.close()
