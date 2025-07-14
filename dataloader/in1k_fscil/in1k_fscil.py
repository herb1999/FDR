import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import clip


class ImageNetFSCIL(Dataset):

    def __init__(self, train=True, index_path=None, wnid_map_file=None):
        """
        ImageNetFSCIL Dataset 类
        - train: 是否是训练集
        - index_path: 训练或测试数据的索引文件 (.csv)
        - index: 训练或测试类别索引（仅用于基类）
        - base_sess: 是否是基类 session
        - autoaug: 是否使用 AutoAugment 数据增强
        - wnid_map_file: WNID (WordNet ID) 映射文件 (统一 1000 类)
        """
        self.train = train
        self.data = []
        self.targets = []
        self.data2label = {}

        # **加载全局 WNID -> Index 映射表**
        self.wnid_to_index = self.load_wnid_map(wnid_map_file)

        # 读取数据索引
        self.data, self.targets = self.SelectfromCsv(index_path)

        # 统一转换 label 为 0~999 索引
        self.targets = [self.wnid_to_index[label] for label in self.targets]

        # 选择数据增强策略
        self.transform = self.get_transform()

    def load_wnid_map(self, wnid_map_file):
        """加载全局 WNID 到 Index 映射 (保证 0~999 一致)"""
        wnid_to_index = {}
        with open(wnid_map_file, "r") as f:
            for line in f.readlines():
                wnid, index = line.strip().split(",")
                wnid_to_index[wnid] = int(index)
        return wnid_to_index

    def get_transform(self):
        """返回数据增强策略"""
        image_size = 224
        return transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def SelectfromCsv(self, csv_path):
        """从 CSV 读取数据"""
        data_tmp = []
        targets_tmp = []
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = os.path.normpath(row['image_path'])
            wnid = row['label']  # WNID (如 n01440764)
            data_tmp.append(img_path)
            targets_tmp.append(wnid)  # 先存 WNID，后续转换索引
            self.data2label[img_path] = wnid
        return data_tmp, targets_tmp


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        """获取单个样本"""
        path, label = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))

        # **确保 label 是 torch.Tensor**
        assert label >= 0, "Negative labels found!"
        assert label < 1000, "Labels exceed class count!"
        return image, torch.tensor(label, dtype=torch.long)


if __name__ == '__main__':
    # 测试数据集
    dataset_root = "/path/to/imagenet_fscil"
    session0_train_csv = os.path.join(dataset_root, "session_0.csv")
    test_csv = os.path.join(dataset_root, "test_session_1.csv")
    wnid_map_file = os.path.join(dataset_root, "imagenet_wnid_map.txt")  # WNID 映射表

    train_dataset = ImageNetFSCIL(root=dataset_root, train=True, index_path=session0_train_csv, base_sess=False,
                                  wnid_map_file=wnid_map_file)
    test_dataset = ImageNetFSCIL(root=dataset_root, train=False, index_path=test_csv, base_sess=False,
                                 wnid_map_file=wnid_map_file)

    print(f"Train Dataset Samples: {len(train_dataset)}")
    print(f"Test Dataset Samples: {len(test_dataset)}")
