import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import numpy as np
from torchvision import transforms
from PIL import Image
import os

class ViCuDataset(Dataset):
    def __init__(self, directory, split='train'):
        """
        初始化数据集类，如果存在 .npy 缓存则直接加载，否则从 .mat 文件中读取并缓存
        """
        self.directory = directory
        npy_files = {
            'current': os.path.join(directory, 'current_data.npy'),
            'labels': os.path.join(directory, 'labels.npy'),
            'train_test_index': os.path.join(directory, 'train_test_index.npy'),
            'video': os.path.join(directory, 'video_data.npy')
        }

        # 如果所有 npy 文件存在，则直接加载
        if all(os.path.exists(path) for path in npy_files.values()):
            print("Loading dataset from .npy files...")
            self.current_data = np.load(npy_files['current'])
            self.labels = np.load(npy_files['labels'])
            self.train_test_index = np.load(npy_files['train_test_index'])
            self.video_data = np.load(npy_files['video'])

        else:
            print("No .npy cache found. Processing from .mat files...")
            self.current_data = []
            self.labels = []
            self.train_test_index = []
            self.video_data = []

            mat_files = get_mat_files_from_directory(directory)

            for mat_file in mat_files:
                with h5py.File(mat_file, 'r') as f:
                    self.current_data.append(np.array(f['current']))
                    self.labels.append(np.array(f['label']))
                    self.train_test_index.append(np.array(f['train_test_index']).flatten())
                    self.video_data.append(np.array(f['video']))

            self.current_data = np.concatenate(self.current_data, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
            self.train_test_index = np.concatenate(self.train_test_index, axis=0)
            self.video_data = np.concatenate(self.video_data, axis=0)
            self.video_data = np.transpose(self.video_data, (0, 2, 3, 1))  # (N, H, W, C)

            # 保存为 .npy 文件
            np.save(npy_files['current'], self.current_data)
            np.save(npy_files['labels'], self.labels)
            np.save(npy_files['train_test_index'], self.train_test_index)
            np.save(npy_files['video'], self.video_data)

        # 归一化电流数据
        self.current_data = (self.current_data - np.mean(self.current_data)) / np.std(self.current_data)

        # 用 train_test_index 区分训练和测试
        self.clip_label_cls = self.train_test_index

        if split == 'train':
            self._filter_data(train=True)
        elif split == 'test':
            self._filter_data(train=False)
        else:
            raise ValueError("split 参数必须是 'train' 或 'test'")

        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _filter_data(self, train=True, normal_test_ratio=0.01, abnormal_test_ratio=0.001):
        """
        过滤训练集或测试集样本：
        - 训练集：仅保留所有正常样本（label 全为 0）
        - 测试集：包含所有异常样本、部分正常样本、部分异常样本

        :param train: 是否为训练集
        :param normal_test_ratio: 测试集中正常样本的比例
        :param abnormal_test_ratio: 测试集中异常样本的比例
        """
        # 找出正常样本（所有像素为 0）
        normal_mask = np.all(self.labels == 0, axis=(1, 2))

        # 异常样本（至少一个像素为 1）
        abnormal_mask = ~normal_mask

        if train:
            selected_mask = normal_mask  # 训练集只包含正常样本
        else:
            # 测试集
            selected_indices = []

            # 选择正常样本
            num_normal = np.sum(normal_mask)
            num_selected_normal = int(num_normal * normal_test_ratio)
            normal_indices = np.where(normal_mask)[0]
            rng = np.random.default_rng(seed=42)  # 保持一致性
            selected_normal_indices = rng.choice(normal_indices, size=num_selected_normal, replace=False)
            selected_indices.extend(selected_normal_indices)

            # 选择异常样本
            num_abnormal = np.sum(abnormal_mask)
            num_selected_abnormal = int(num_abnormal * abnormal_test_ratio)
            abnormal_indices = np.where(abnormal_mask)[0]
            selected_abnormal_indices = rng.choice(abnormal_indices, size=num_selected_abnormal, replace=False)
            selected_indices.extend(selected_abnormal_indices)

            # 组合正常和异常样本的索引
            selected_mask = np.zeros(len(self.labels), dtype=bool)
            selected_mask[selected_indices] = True

        # 应用掩码
        self.current_data = self.current_data[selected_mask]
        self.labels = self.labels[selected_mask]
        self.video_data = self.video_data[selected_mask]
        self.train_test_index = self.train_test_index[selected_mask]

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
        current_sample = torch.tensor(self.current_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        video_sample = self.video_data[idx]
        video_sample = np.clip(video_sample, 0, 255).astype(np.uint8)
        video_sample = Image.fromarray(video_sample)
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
    
    # 获取训练数据
    train_dataset = ViCuDataset(directory=directory, split='train')

    # 获取测试数据
    test_dataset = ViCuDataset(directory=directory, split='test')

    # 打印训练集和测试集的总样本数量
    print(f"Total number of samples in the training set: {len(train_dataset)}")
    print(f"Total number of samples in the testing set: {len(test_dataset)}")

    # 使用 DataLoader 加载数据
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
