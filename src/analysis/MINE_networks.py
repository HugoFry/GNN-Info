from dataclasses import dataclass, field
from torch import nn
from config import MINE_config, MLP_config
from MLP import MLP_network
import torch

class MINE_networks():
    def __init__(self, config: MINE_config):
        self.config = config
        self.networks = {'input network': [], 'label network': []}
        configs = config.get_individual_configs()
        for config in configs['input config']:
            network = MLP_network(config)
            self.networks['input network'].append(network)
            
        for config in configs['label config']:
            network = MLP_network(config)
            self.networks['label network'].append(network)

    
    def __getitem__(self, index: int):
        return {'input network': self.networks['input network'][index], 'label network': self.networks['label network'][index]}
    
    def __len__(self):
        return len(self.config.GNN_config.dimensions)-1