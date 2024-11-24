import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ODIRDataset(Dataset):
    def __init__(self, folder_path):
        """
        Args:
            folder_path (str): 数据集文件夹路径（train/test/val）
        """
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
        # 定义图像预处理方法，包括调整到 224x224
        self.resize = transforms.Resize((224, 224))  # 调整大小

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载 npz 文件
        data = np.load(self.files[idx])

        # 获取所需的键值
        slo_fundus = data['slo_fundus']  # 图像 (200, 200, 3)
        male = data['male']              # 性别 (0 或 1)
        dr_class = data['dr_class']      # 是否患病 (0 或 1)

        # 预处理：将图像转换为 PyTorch 张量
        slo_fundus = torch.tensor(slo_fundus, dtype=torch.float32).permute(2, 0, 1) / 255.0  # 转换为 (C, H, W)，归一化
        slo_fundus = self.resize(slo_fundus)
        male = torch.tensor(male, dtype=torch.long)  # 转换为张量
        dr_class = torch.tensor(dr_class, dtype=torch.long)  # 转换为张量（分类标签需要 long）

        # 返回数据
        return slo_fundus, male, dr_class


def load_datasets(base_path, batch_size=32):
    """
    加载 train、val 和 test 数据集，并返回 DataLoader。
    Args:
        base_path (str): 数据集的根目录，包含 train、val、test 文件夹
        batch_size (int): DataLoader 的批量大小
    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    train_dataset = ODIRDataset(os.path.join(base_path, 'train'))
    val_dataset = ODIRDataset(os.path.join(base_path, 'val'))
    test_dataset = ODIRDataset(os.path.join(base_path, 'test'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 加载数据
    train_loader, val_loader, test_loader = load_datasets('./dataset', batch_size=32)

    # 遍历数据
    for images, genders, labels in train_loader:
        print("Images shape:", images.shape)  # (batch_size, 3, 200, 200)
        print("Genders:", genders)  # (batch_size,)
        print("Labels:", labels)  # (batch_size,)
        break