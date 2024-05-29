from dataclasses import dataclass, field
from torch import nn
import torch


@dataclass
class MINE_config():
    dimensions: list[int] = field(default_factory=lambda: [100, 200, 100, 1])
    dropout_prob: float = 0.5
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    weight_decay: float = 0.01
    optimizer: str = 'Adam'
    

class MI_network(nn.module):
    def __init__(self, config: MINE_config):
        super(MI_network, self).__init__()
        self.config = config
        layers = []
        for i in range(len(config.dimensions)-1):
            layers.append(nn.Linear(config.dimensions[i], config.dimensions[i+1]))
            
            if i < len(config.dimensions) - 2:  # No activation/batchnorm/dropout on the final layer
                layers.append(nn.BatchNorm1d(config.dimensions[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout_prob))
                
        self.layers = nn.Sequential(*layers)
        
        if config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
    def forward(self, x):
        return self.layers[x]