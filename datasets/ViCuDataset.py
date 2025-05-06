import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import numpy as np
from torchvision import transforms
from PIL import Image
import os

class ViCuDataset(Dataset):
    def __init__(self, directory, split='train', transform=None):
        """
        初始化数据集类，合并多个 .mat 文件的数据，并根据split选择训练或测试数据
        
        :param mat_files: 包含多个 .mat 文件路径的列表
        :param split: 数据集的类型，'train' 或 'test'
        :param transform: 可选的图像数据变换
        """
        self.current_data = []
        self.labels = []
        self.train_test_index = []
        self.video_data = []

        mat_files = get_mat_files_from_directory(directory)
        
        # 逐个读取每个 .mat 文件
        for mat_file in mat_files:
            with h5py.File(mat_file, 'r') as f:
                # 提取数据并合并
                self.current_data.append(np.array(f['current']))
                self.labels.append(np.array(f['label']))
                self.train_test_index.append(np.array(f['train_test_index']).flatten())
                self.video_data.append(np.array(f['video']))
        
        # 将数据合并成一个大数组
        self.current_data = np.concatenate(self.current_data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.train_test_index = np.concatenate(self.train_test_index, axis=0)
        self.video_data = np.concatenate(self.video_data, axis=0)

        # self.video_data = np.transpose(self.video_data, axes=(0, 3, 2, 1))
        # self.labels = np.transpose(self.labels, axes=(0, 2, 1))
        self.clip_label_cls = np.max(self.labels)

        # 标准化
        self.current_data = (self.current_data - np.mean(self.current_data)) / np.std(self.current_data)
        
        # 根据 split 筛选数据（'train' 或 'test'）
        if split == 'train':
            self._filter_data(train=True)
        elif split == 'test':
            self._filter_data(train=False)
        else:
            raise ValueError("split 参数必须是 'train' 或 'test'")

        # 可选的图像变换
        self.transform = transform

    def _filter_data(self, train=True):
        """
        根据 train_test_index 过滤训练集或测试集数据
        :param train: True 返回训练集数据，False 返回测试集数据
        """
        if train:
            mask = self.train_test_index == 0  # 假设 0 表示训练集
        else:
            mask = self.train_test_index == 1  # 假设 1 表示测试集
        
        self.current_data = self.current_data[mask]
        self.labels = self.labels[mask]
        self.video_data = self.video_data[mask]
        self.train_test_index = self.train_test_index[mask]

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.current_data)
    
    def __getitem__(self, idx):
        """
        根据索引返回一对数据，包含电流数据、视频数据和标签。
        
        :param idx: 数据的索引
        :return: 一对电流数据、视频数据和标签
        """
        current_sample = self.current_data[idx]  # 电流数据 (3,)
        label = self.labels[idx]  # 标签
        video_sample = self.video_data[idx]  # 视频数据 (64, 64, 3)

        # 如果有图像变换，则应用
        if self.transform:
            video_sample = Image.fromarray(video_sample.astype(np.uint8))
            video_sample = self.transform(video_sample)

        return current_sample, video_sample, label

# 获取指定目录下所有的 .mat 文件
def get_mat_files_from_directory(directory):
    mat_files = []
    for file in os.listdir(directory):  # 遍历指定目录
        if file.endswith('.mat'):
            mat_files.append(os.path.join(directory, file))  # 获取完整路径
    return mat_files

# 示例：使用该类读取指定目录下所有的 .mat 文件并返回合并后的数据
if __name__ == "__main__":
    # 指定目录路径
    directory = "data/FMFBenchmarkV1"  # 替换为你指定的目录路径
    
    mat_files = get_mat_files_from_directory(directory)  # 获取指定目录下的所有 .mat 文件
    
    # 如果需要对图像进行标准化处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 获取训练数据
    train_dataset = ViCuDataset(mat_files=mat_files, split='train', transform=transform)

    # 获取测试数据
    test_dataset = ViCuDataset(mat_files=mat_files, split='test', transform=transform)

    # 打印训练集和测试集的总样本数量
    print(f"Total number of samples in the training set: {len(train_dataset)}")
    print(f"Total number of samples in the testing set: {len(test_dataset)}")

    # 使用 DataLoader 加载数据
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
