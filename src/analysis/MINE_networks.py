from dataclasses import dataclass, field
from torch import nn
from config import MINE_config, MLP_config
from MLP import MLP_network
import torch

class MINE_networks():
    def __init__(self, config: MINE_config):
        self.config = config
        self.networks = {'input': [], 'label': []}
        configs = config.get_individual_configs()
        
        for network_type in ['input', 'label']:
            for config in configs[network_type]:
                network = MLP_network(config)
                self.networks[network_type].append(network)

    
    def __getitem__(self, network_type: str):
        """
        Returns a list of networks. The list indices the layers of the GNN.
        """
        return self.networks[network_type]
    
    def __len__(self):
        return len(self.config.GNN_config.dimensions)-1
    
    def loss_function(self, joint_output, marginal_output):
        """
        Loss function given in the MINE paper.
        The form of the loss funciton is given by the Donsker-Varadhan representation.
        """
        joint_loss = torch.mean(joint_output, dim = 0)
        marginal_loss = torch.log(torch.mean(torch.exp(marginal_output), dim = 0))
        loss = joint_loss - marginal_loss
        return loss