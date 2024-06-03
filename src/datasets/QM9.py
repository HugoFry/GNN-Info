import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

def get_dataset(path='./data/QM9'):
    dataset = QM9(path) # This should automatically check whether the dataset already exists before downloading.
    return dataset
