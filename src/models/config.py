from torch import nn
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch

@dataclass
class GNN_config(ABC):
    GNN_layer: str = 'GCN'
    dataset: str = 'QM9'
    dimensions: list[int] = field(default_factory = lambda: [10,100,100,100,10]) # Input and output dimensions should be inferred from the dataset.
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    weight_decay: float = 0.01 #Not sure if I want to use weight decay in graph neural networks.
    optimizer: str = 'adam'
    head: str = 'linear'
    head_output_dimension: int = 10 # Note this is the output dimension of the GNN only when head is not None, otherwise it is dimensions[-1]
    activation_function: str = "relu"
    
    def __repr__(self):
        attrs = (f"{k} = {v!r}" for k, v in self.__dict__.items())
        attr_str = "\n".join(attrs)
        return f"GNN configuration:\n\n{attr_str}"
    
    def __post_init__(self):
        loss_functions = {
            'QM9': 'log-cosh',
            'reddit': 'cross-entropy',
            }
        self.loss_function = loss_functions[self.dataset]