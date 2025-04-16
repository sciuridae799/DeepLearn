import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import random_split


def dataloader(batch_size=64, root='./01-pytorch/1.5mnist/data', split=(0.7, 0.15, 0.15)):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), 
    ])

    # 加载MNIST数据集
    dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)

    dataset_size = len(dataset)

    # 数据集分割
    train_size = int(split[0] * dataset_size)
    val_size = int(split[1] * dataset_size)
    test_size = dataset_size - train_size - val_size

    # 使用 random_split 将数据集划分为训练集、验证集、测试集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 使用 MNIST 数据集
    root = './01-pytorch/1.5mnist/data'
    train_loader, val_loader, test_loader = dataloader(root=root)

    for images, labels in train_loader:
        print(images.shape)  # 输出图像的形状 (batch_size, 1, 28, 28)
        print(labels.shape)  # 输出标签的形状 (batch_size,)
        break