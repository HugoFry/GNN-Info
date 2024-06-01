from dataclasses import dataclass, field
from torch import nn
import torch


@dataclass
class MINE_config():
    GNN_model: str = 'GCN'
    GNN_dataset: str = 'QM9'
    activations_dataset_size: int = 50_000
    activation_function: str = 'relu'
    dimensions: list[int] = field(default_factory=lambda: [100, 200, 100, 1])
    dropout_prob: float = 0.5
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    weight_decay: float = 0.01
    optimizer: str = 'Adam'
    
class MLP_layer(nn.module):
    def __init__(self, config: MINE_config, input_dimensions: int, output_dimensions: int):
        super(MLP_layer, self).__init__()
        self.linear = nn.Linear(input_dimensions, output_dimensions)
        self.batch_norm = nn.BatchNorm1d(output_dimensions)
        if config.activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif config.activation_function == 'sigmoid':
            self.activation_function = nn.Sigmoid()
        else:
            raise Exception(f'Unrecognised activation function {config.activation_function}')
        self.dropout = nn.Dropout(config.dropout_prob)
        
        #Initialise the weights and biases.
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)
        x = self.dropout(x)
        return x
    
class linear_layer(nn.module):
    def __init__(self, input_dimensions: int, output_dimensions: int):
        super(linear_layer, self).__init__()
        self.linear = nn.Linear(input_dimensions, output_dimensions)
        
        #Initialise the weights and biases.
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        x = self.linear(x)
        return x


class MI_network(nn.module):
    def __init__(self, config: MINE_config):
        super(MI_network, self).__init__()
        self.config = config
        layers = []
        
        #Define all but the last layer
        for i in range(len(config.dimensions)-2):
            layers.append(MLP_layer(config, config.dimensions[i], config.dimensions[i+1]))
        
        #Just a linear layer for the output layer
        layers.append(linear_layer(config.dimensions[-2], config.dimensions[-1]))
                
        self.layers = nn.Sequential(*layers)
        
        if config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            raise Exception(f'The optimizer {config.optimizer} is not recognised')
        
    def forward(self, x):
        return self.layers[x]