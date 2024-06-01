import torch
import sys
from torch import nn
from torch.utils.data import DataLoader

def train_GNN(GNN_model, GNN_dataset, MINE_model):
    print(GNN_model.config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = GNN_model.config.epochs
    
    GNN_model.to(device)
    GNN_dataloader = DataLoader(GNN_dataset, batch_size=GNN_model.config.batch_size, shuffle=True)
    
    for epoch in epochs:
        epoch_loss = 0
        GNN_model.train() # Need to set to train at the start of each epoch as the model is analysed at the end of the epoch
        
        for batch_data in GNN_dataloader:
            batch_data.to(device)
            
            GNN_model.optimizer.zero_grad()
            
            output = GNN_model(batch_data.x)
            
            loss = None # Use batch_data.y or something like that.
            
            loss.backward()
            
            GNN_model.optimizer.step()
            
            epoch_loss += loss.item()
            
        GNN_model.eval()
        
#####
#WRONG I need to have several MINE models. Not just one model. But I want only one activations dataset.
#####
def train_MINE(GNN_model, GNN_dataset, MINE_model):
    activations_dataset = activations_dataset(GNN_model, GNN_dataset, MINE_model.config.activations_dataset_size)
    pass