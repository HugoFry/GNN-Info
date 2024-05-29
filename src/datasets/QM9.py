import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

def get_dataset(path='./data/QM9'):
    dataset = QM9(path) # This should automatically check whether the dataset already exists before downloading.
    return dataset

def get_dataloader(dataset, batch_size=32, shuffle=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
