import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
from ..analysis.activations_dataset import activations_dataset
from ..analysis.MINE_networks import MINE_networks

def train_GNN(GNN_model, GNN_dataset, MINE_models: list):
    print(GNN_model.config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = GNN_model.config.epochs
    
    GNN_model.to(device)
    for layer in range(len(MINE_models)):
        MINE_models[layer]['input'].to(device)
        MINE_models[layer]['label'].to(device)
        
    GNN_dataloader = DataLoader(GNN_dataset, batch_size=GNN_model.config.batch_size, shuffle=True)
    
    for epoch in epochs:
        epoch_loss = 0
        GNN_model.train() # Need to set to train at the start of each epoch as the model is analysed in evals mode at the end of the epoch
        
        for batch_data in GNN_dataloader:
            batch_data.to(device)
            
            GNN_model.optimizer.zero_grad()
            
            output = GNN_model(batch_data.x)
            
            #GNN_model.config.loss_function
            loss = GNN_model.loss_function(output, batch_data.y) # Use batch_data.y or something like that. Loss will depend on dataset probably? Need to think about that.
            
            loss.backward()
            
            GNN_model.optimizer.step()
            
            epoch_loss += loss.item()
            #Log to WnB here
            
        GNN_model.eval()
        train_MINE(GNN_model, GNN_dataset, MINE_models, device)
        #Save MINE values!!!
        #Log to WnB here too
        

def train_MINE(GNN_model, GNN_dataset, MINE_models: MINE_networks, device):
    """
    To do:
        Think about a test/train split
        Think about logging to WnB. What should I log?
    """
    MINE_dataset = activations_dataset(GNN_model, GNN_dataset, MINE_models.config.activations_dataset_size)
    MINE_iterable_dataloader = iter(DataLoader(MINE_dataset, batch_size = MINE_models.config.batch_size, shuffle = True))
    
    for layer in range(len(MINE_models)):
        for network_type in ['input', 'label']:
            MINE_models[network_type][layer].train()
    
    epoch = 0
    while epoch < MINE_models.config.epochs:
        #Draw two batches of data from the dataset to create the marginal and joint distributions
        try:
            batch_a = next(MINE_iterable_dataloader)
            batch_b = next(MINE_iterable_dataloader)
        except StopIteration:
            epoch +=1
            MINE_iterable_dataloader = iter(DataLoader(MINE_dataset, batch_size = MINE_models.config.batch_size, shuffle = True))
            batch_a = next(MINE_iterable_dataloader)
            batch_b = next(MINE_iterable_dataloader)
        
        joint_data = {
            'input': [torch.cat((batch_a[0], batch_a[layer+1]), dim = -1) for layer in len(batch_a)-2],
            'label': [torch.cat((batch_a[-1], batch_a[layer+1]), dim = -1) for layer in len(batch_a)-2],
            }
        
        marginal_data = {
            'input': [torch.cat((batch_a[0], batch_b[layer+1]), dim = -1) for layer in len(batch_a)-2],
            'label': [torch.cat((batch_a[-1], batch_b[layer+1]), dim = -1) for layer in len(batch_a)-2],
            }
        
        for layer in range(len(MINE_models)):
            for network_type in ['input', 'label']:
                model = MINE_models[network_type][layer]
                model.optimizer.zero_grad()
                joint_output = model(joint_data[network_type][layer])
                marginal_output = model(marginal_data[network_type][layer])
                loss = MINE_models.loss_function(joint_output, marginal_output)
                loss.backward()
                model.optimizer.step()