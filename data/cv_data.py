import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST, SVHN, CIFAR10
import torchvision.transforms as transforms

_CV_DATASETS = {}

def _add_dataset(dataset_fn):
    _CV_DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn


class FlattenTransform:
    def __call__(self, x):
        return x.view(-1)
    
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    FlattenTransform()
])

transform_svhn = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomRotation(10),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4453], std=[0.1970]),
    #transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    FlattenTransform()
])

transform_cifar_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

@_add_dataset
def mnist(root):
    train_dataset = MNIST(root=root, train=True, download=True, transform=transform_mnist)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform_mnist)

    return train_dataset, test_dataset

@_add_dataset
def svhn(root):
    root = os.path.join(root, 'svhn')
    train_dataset = SVHN(root=root, split='train', download=True, transform=transform_svhn)
    test_dataset = SVHN(root=root, split='test', download=True, transform=transform_svhn)

    return train_dataset, test_dataset

@_add_dataset
def cifar10(root):
    root = os.path.join(root, 'cifar10')
    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform_cifar_train)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform_cifar_test)

    return train_dataset, test_dataset


def get_cv_dataset(dataset_name, root=None):
    if root is None:
        root = 'data'
    return _CV_DATASETS[dataset_name](root)