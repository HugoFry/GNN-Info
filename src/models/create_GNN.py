from torch import nn
from config import GNN_config
from GCN import GCN
from GAT import GAT
from MPNN import MPNN
import torch

def create_GNN(config: GNN_config):
    if config.GNN_layer == "GCN":
        GNN = GCN(config)
    elif config.GNN_layer == "GAT":
        GNN = GAT(config)
    elif config.GNN_layer == "MPNN":
        GNN = MPNN(config)
    return GNN