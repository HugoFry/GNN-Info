from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn
from dataclasses import dataclass, field
from contextlib import contextmanager
from config import GNN_config
import torch

class GCN_layer(nn.Module):
    def __init__(self,  config: GNN_config, input_dimension: int, output_dimension: int):
        super(GCN_layer, self).__init__()
        self.GCN = GCNConv(input_dimension, output_dimension)
        if config.activation_function == "relu":
                self.activation_funciton = nn.ReLu()
        elif config.activation_function == "sigmoid":
            self.activation_funciton = nn.Sigmoid()
        else:
            raise Exception(f"Unrecognised activation function {config.activation_function}.")
        
    def forward(self, x, edge_index):
        x = self.GCN(x, edge_index)
        x = self.activation_funciton(x)
        return x
    
class linear_layer(nn.Module):
    def __init__(self,  config: GNN_config, input_dimension: int, output_dimension: int):
        super(GCN_layer, self).__init__()
        self.linear = Linear(input_dimension, output_dimension)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        x = self.Linear(x)
        return x
    
class softmax_layer(nn.Module):
    def __init__(self,  config: GNN_config, input_dimension: int, output_dimension: int):
        super(GCN_layer, self).__init__()
        self.linear = Linear(input_dimension, output_dimension)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, x):
        x = self.Linear(x)
        x = self.softmax(x)
        return x
        

class GCN(nn.Module):
    def __init__(self, config: GNN_config):
        super(GCN, self).__init__()
        self.config = config
        
        layers = []
        for i in range(len(config.dimensions)-1):
            layers.append(GCN_layer(config, config.dimensions[i], config.dimensions[i+1]))    
        self.layers = nn.Sequential(*layers)
        
        if config.head == "linear":
            self.head = linear_layer(config, config.dimesions[-1], config.head_output_dimension)
        elif config.head == "softmax":
            self.head = softmax_layer(config, config.dimensions[i], config.dimensions[i+1])
        else:
            config.head = nn.Identity()
            
        if config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            raise Exception(f'The optimizer {config.optimizer} is not recognised')

    def forward(self, x, edge_index):
        x = self.layers(x, edge_index)
        x = self.head(x)
        return x
    
    def run_with_cache(self, *args, **kwargs):
        """
        Hooks the model to chache the activations and then perfroms the forward pass. Does this with a generator thing. Returns the model output and the cached activations as a list.
        """
        # Store the activations as a list of tensors.
        cached_activations = [None for _ in len(self.layers)]
        
        #If there is a non-trivial head, append an additional element to the list
        if self.config.head is not None:
            cached_activations.append(None)
            
        with self.hooked_model(cached_activations):
            output = self.forward(*args, **kwargs)
            
        return output, cached_activations
    
    def create_hook_function(self, cached_activations: list, layer: int):
        def hook_function(model, input, output):
            cached_activations[layer] = output.detach().clone()
        return hook_function
    
    @contextmanager
    def hooked_model(self, cached_activations: list):
        try:
            #Hook the model here before the forward pass.
            handles = []
            for layer, module in enumerate(self.layers):
                hook_fn = self.create_hook_function(cached_activations, layer)
                handle = module.register_forward_hook(hook_fn)
                handles.append(handle)
                
            if self.config.head is not None:
                hook_fn = self.create_hook_function(cached_activations, -1)
                handle = self.head.register_forward_hook(hook_fn)
                handles.append(handle)
                
            yield
        finally:
            #Remove the hooks here.
            for handle in handles:
                handle.detach()
    
