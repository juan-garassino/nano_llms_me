import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import List, Dict
import os

class RobustMNIST(datasets.MNIST):
    """Custom MNIST that avoids calling .numpy() in __getitem__ to bypass environment errors."""
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # img is uint8 tensor (28, 28)
        # Manually convert to float and normalize to [-1, 1]
        # This replaces ToTensor() and Normalize((0.5,), (0.5,))
        img = img.float().unsqueeze(0) / 255.0
        img = (img - 0.5) / 0.5
        
        return img, target

class RobustFashionMNIST(datasets.FashionMNIST):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = img.float().unsqueeze(0) / 255.0
        img = (img - 0.5) / 0.5
        return img, target

class RobustKMNIST(datasets.KMNIST):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = img.float().unsqueeze(0) / 255.0
        img = (img - 0.5) / 0.5
        return img, target

def load_dataset(name: str, root: str, train: bool, transform):
    name = name.lower()
    # Note: We ignore the 'transform' argument because we hardcode the normalization 
    # in the Robust classes to ensure we don't trigger PIL/numpy conversions.
    
    if name in ["mnist"]:
        ds = RobustMNIST(root, train=train, download=True)
    elif name in ["fashionmnist", "fashion"]:
        ds = RobustFashionMNIST(root, train=train, download=True)
    elif name in ["kmnist"]:
        ds = RobustKMNIST(root, train=train, download=True)
    else:
        ds = RobustMNIST(root, train=train, download=True)
    return ds, 10

def get_multimnist_loaders(dataset_names: List[str], batch_size: int, root="./data", num_workers=0):
    # Transform is unused but kept for signature compatibility if needed later
    transform = None 
    loaders = {}
    class_counts = {}
    
    for name in dataset_names:
        train_ds, n_cls = load_dataset(name, root, True, transform)
        test_ds, _ = load_dataset(name, root, False, transform)
        class_counts[name] = n_cls
        per_bs = max(1, batch_size // len(dataset_names))
        loaders[name] = {
            "train_loader": DataLoader(train_ds, batch_size=per_bs, shuffle=True, num_workers=num_workers),
            "test_loader": DataLoader(test_ds, batch_size=per_bs, shuffle=False)
        }
    return loaders, class_counts
