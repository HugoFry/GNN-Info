from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn
from dataclasses import dataclass, field
import torch

@dataclass
class GCN_config():
    dimensions: list[int] = field(default_factory = lambda: [10,100,100,100,10])
    dropout_prob: float = 0.5
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    weight_decay: float = 0.01
    optimizer: str = 'Adam'
    head: str = 'linear'
    head_output_dimension: int = 10
        

class GCN(nn.Module):
    def __init__(self, config: GCN_config):
        super(GCN, self).__init__()
        self.config = config
        layers = []
        for i in range(len(config.dimensions)-1):
            layers.append(GCNConv(config.dimensions[i], config.dimensions[i+1]))
            
            if i < len(config.dimensions) - 2:  # No activation/batchnorm/dropout on the final layer
                layers.append(nn.ReLu())
                layers.append(nn.Dropout(config.dropout_prob))
                
        self.layers = nn.Sequential(*layers)
        if self.head == "linear":
            self.head = Linear(config.dimesions[-1], config.head_output_dimension)
        else:
            self.head = nn.Identity()
            
        if config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def forward(self, x, edge_index):
        for layer in self.layers:
            if isinstance(layer, GCNConv): # Check whether we need to pass in the adjacency matrix.
                x = layer(x, edge_index)
            else:
                x = layer(x)
                
        x = self.head(x)
        return x