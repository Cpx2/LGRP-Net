""" Additional utility functions. """
import matplotlib
matplotlib.use('pdf')
import os
import time
import pprint
import torch
import numpy as np
import torch.nn.functional as F


def ensure_path(path):
    """The function to make log path.
    Args:
      path: the generated saving path.
    """
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

class Averager():
    """The class to calculate the average."""
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc(logits, label):
    """The function to calculate the accuracy.
    Args:
      logits: input logits.
      label: ground truth labels.
    Return:
      The output accuracy.
    """
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()

class Timer():
    """The class for timer."""
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """The function to calculate the confidence interval.
    Args:
      data: input records
    Return:
      m: mean value
      pm: confidence interval.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, v2.T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def get_cos_similar_matrix_loop(v1, v2):
    simmat = np.zeros((v1.shape[0], v2.shape[0]))
    for k in range(v1.shape[0]):
        for i in range(v2.shape[0]):
            num = np.dot(v1[k], v2[i])
            denom = np.linalg.norm(v1[k]) * np.linalg.norm(v2[i])
            res = num / denom
            simmat[k, i] = res
    return 0.5 + 0.5 * simmat


def np_proto(feat, label, way):
    """修复：标签重映射，避免索引越界"""
    feat_proto = np.zeros((way, feat.shape[1]))
    
    # 获取唯一标签并重新映射到 0~len(unique)-1
    unique_labels = np.unique(label)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_label = np.array([label_map[l] for l in label])
    
    # 只处理实际存在的标签，且不超过way数
    for idx, lb in enumerate(unique_labels):
        if idx >= way:  # 防止索引越界
            break
        ds = np.where(remapped_label == idx)[0]
        feat_ = feat[ds]
        feat_proto[idx] = np.mean(feat_, axis=0)
    
    return feat_proto


def tc_proto(feat, label, way):
    """修复：标签重映射，避免索引越界"""
    feat_proto = torch.zeros(way, feat.size(1)).to(feat.device)
    
    # 获取唯一标签并重新映射到 0~len(unique)-1
    unique_labels = torch.unique(label)
    label_map = {old.item(): new for new, old in enumerate(unique_labels)}
    remapped_label = torch.tensor([label_map[l.item()] for l in label]).to(label.device)
    
    # 只处理实际存在的标签，且不超过way数
    for idx, lb in enumerate(unique_labels):
        if idx >= way:  # 防止索引越界
            break
        ds = torch.where(remapped_label == idx)[0]
        feat_ = feat[ds]
        feat_proto[idx] = torch.mean(feat_, dim=0)
    
    return feat_proto


def updateproto(Xs, ys, cls_center, way):
    """assign labels for the clusters（修复版本）"""
    # 标签重映射
    unique_labels = np.unique(ys)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    ys_remapped = np.array([label_map[l] for l in ys])
    
    proto = np_proto(Xs, ys_remapped, way)
    
    # 适配聚类中心维度
    if cls_center.shape[0] != way:
        if cls_center.shape[0] > way:
            cls_center = cls_center[:way]
        else:
            padding = np.zeros((way - cls_center.shape[0], cls_center.shape[1]))
            cls_center = np.concatenate([cls_center, padding], axis=0)
    
    dist = ((proto[:, np.newaxis, :]-cls_center[np.newaxis, :, :])**2).sum(2)
    id = dist.argmin(1)
    feat_proto = np.zeros((way, Xs.shape[1]))
    
    for i in range(min(len(unique_labels), way)):
        feat_proto[i] = (cls_center[id[i]] + proto[i])/2
    
    return feat_proto


def compactness_loss(gen_feat, gen_label, proto, supp_label):
    """修复：使用重映射后的标签计算损失 + MSELoss 参数更新"""
    # 修复：MSELoss 参数过时
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = 0

    # 重映射支持集标签
    unique_supp_labels = torch.unique(supp_label)
    supp_label_map = {old.item(): new for new, old in enumerate(unique_supp_labels)}
    supp_label_remapped = torch.tensor([supp_label_map[l.item()] for l in supp_label]).to(supp_label.device)

    # 只计算实际存在的类别损失
    for idx, lb in enumerate(unique_supp_labels):
        if idx >= proto.shape[0]:  # 防止索引越界
            break
        id = torch.where(gen_label == lb)[0]
        if len(id) > 0:  # 确保有对应样本
            loss = loss + loss_fn(gen_feat[id], proto[idx])

    return loss


def euclidean_metric(a, b):
    """修复：适配不同维度的输入"""
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits