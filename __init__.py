import torch

# 替换为你的文件路径
file_path = "D:/python/PyCharm 2021.2.2/FewShot/center/center_MiniImageNet_swin.pth"

try:
    data = torch.load(file_path, map_location='cpu')
    print("文件加载成功！包含的键：", data.keys())
except Exception as e:
    print("文件加载失败：", e)

import torch

# 替换为你的文件路径
file_path = "./checkpoints/mini-resnet12.pth"

data = torch.load(file_path, map_location='cpu')
print("预训练文件中的键：", data.keys())