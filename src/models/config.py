from torch import nn
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch

@dataclass
class GNN_config(ABC):
    GNN_layer: str = 'GCN'
    dataset: str = 'QM9'
    dimensions: list[int] = field(default_factory = lambda: [10,100,100,100,10])
    dropout_prob: float = 0.5
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    weight_decay: float = 0.01
    optimizer: str = 'Adam'
    head: str = 'linear'
    head_output_dimension: int = 10 # Note this is the output dimension of the GNN only when head is not None, otherwise it is dimensions[-1]
    activation_function: str = "relu"