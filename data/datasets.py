import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# ===================== 核心修改：配置本地绝对路径 =====================
# 替换为你生成的MiniImageNet根目录（关键：用绝对路径避免相对路径错误）
MINI_IMAGENET_ROOT = r"D:\python\PyCharm 2021.2.2\FewShot\datasets\MiniImageNet"
# 图片根目录（拼接绝对路径）
ROOT = os.path.join(MINI_IMAGENET_ROOT, "images")
# CSV文件根目录（你的train/val/test.csv所在路径）
CSV_ROOT = MINI_IMAGENET_ROOT


class my_dataset(Dataset):
    def __init__(self, csv_path='train.csv', img_size=84, have_root=True, transform=None):
        """
        数据集加载类修改说明：
        1. csv_path：支持传入CSV文件名（如'train.csv'）或绝对路径
        2. have_root：控制是否使用拼接后的绝对路径加载图片
        3. 修复路径分割、拼接的跨平台问题（兼容Windows的\和Linux的/）
        """
        # 第一步：处理CSV路径，确保找到正确的CSV文件
        if not os.path.isabs(csv_path):
            # 若传入的是相对路径（如'train.csv'），拼接为绝对路径
            csv_full_path = os.path.join(CSV_ROOT, csv_path)
        else:
            csv_full_path = csv_path

        # 检查CSV文件是否存在
        if not os.path.exists(csv_full_path):
            raise FileNotFoundError(f"CSV文件不存在：{csv_full_path}")
        print(f"正在加载CSV文件：{csv_full_path}")

        # 第二步：读取CSV文件并处理行数据
        with open(csv_full_path, 'r', encoding='utf-8') as f:
            lines1 = [x.strip() for x in f.readlines()][1:]  # 跳过表头

        # 初始化变量
        data = []  # 图片路径列表
        label = []  # 标签索引列表
        lb = -1  # 标签计数器
        self.wnids = []  # 存储所有WNID
        self.label2catname = {}  # 标签索引到WNID的映射

        # 第三步：处理每一行的图片路径和WNID
        for l in lines1:
            if not l:  # 跳过空行
                continue
            try:
                # 分割图片路径和WNID（兼容CSV中,分隔的格式）
                img_path, wnid = l.split(',', 1)
                img_path = img_path.strip()
                wnid = wnid.strip()
            except ValueError:
                print(f"警告：CSV行格式错误，跳过该行：{l}")
                continue

            # 处理图片路径：确保是本地绝对路径
            if have_root:
                # 若CSV中的路径是相对路径（如images/n01443537/xxx.jpg），拼接为绝对路径
                if not os.path.isabs(img_path):
                    img_abs_path = os.path.join(MINI_IMAGENET_ROOT, img_path)
                else:
                    img_abs_path = img_path
            else:
                # 直接使用CSV中的路径（若为绝对路径）
                img_abs_path = img_path

            # 检查图片文件是否存在（可选：用于调试）
            if not os.path.exists(img_abs_path):
                print(f"警告：图片文件不存在，跳过：{img_abs_path}")
                continue

            # 标签映射：WNID到连续整数索引
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                self.label2catname[lb] = wnid

            data.append(img_abs_path)
            label.append(lb)

        # 赋值给实例变量
        self.data = data
        self.label = label
        self.img_size = img_size

        # 第四步：设置数据变换（保持原逻辑不变）
        if transform is None:
            if 'train' in csv_path or 'source' in csv_path:
                self.transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                        np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
                    )
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                        np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
                    )
                ])
        else:
            self.transform = transform

        # 打印加载信息（用于验证）
        print(f"数据集加载完成：共 {len(self.data)} 个样本，{len(self.wnids)} 个类别")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        try:
            path, label = self.data[i], self.label[i]
            # 打开图片并转换为RGB（避免灰度图报错）
            image = Image.open(path).convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            # 异常处理：避免单张图片加载失败导致程序崩溃
            print(f"加载图片失败：{path}，错误：{e}")
            # 返回空白图片和标签（兜底）
            blank_img = Image.new('RGB', (self.img_size, self.img_size), color='white')
            return self.transform(blank_img), label


# ===================== 测试代码（可选：验证数据集加载） =====================
if __name__ == "__main__":
    # 测试加载训练集
    train_dataset = my_dataset(csv_path='train.csv', img_size=84)
    print(f"训练集样本数：{len(train_dataset)}")
    print(f"类别数：{len(train_dataset.wnids)}")
    print(f"第一个样本的标签索引：{train_dataset.label[0]}，对应WNID：{train_dataset.label2catname[train_dataset.label[0]]}")

    # 测试加载验证集
    val_dataset = my_dataset(csv_path='val.csv', img_size=84)
    print(f"验证集样本数：{len(val_dataset)}")

    # 测试加载测试集
    test_dataset = my_dataset(csv_path='test.csv', img_size=84)
    print(f"测试集样本数：{len(test_dataset)}")