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
        MINE_models[layer]['input network'].to(device)
        MINE_models[layer]['label network'].to(device)
        
    GNN_dataloader = DataLoader(GNN_dataset, batch_size=GNN_model.config.batch_size, shuffle=True)
    
    for epoch in epochs:
        epoch_loss = 0
        GNN_model.train() # Need to set to train at the start of each epoch as the model is analysed in evals mode at the end of the epoch
        
        for batch_data in GNN_dataloader:
            batch_data.to(device)
            
            GNN_model.optimizer.zero_grad()
            
            output = GNN_model(batch_data.x)
            
            loss = None # Use batch_data.y or something like that. Loss will depend on dataset probably? Need to think about that.
            
            loss.backward()
            
            GNN_model.optimizer.step()
            
            epoch_loss += loss.item()
            #Log to WnB here
            
        GNN_model.eval()
        train_MINE(GNN_model, GNN_dataset, MINE_models, device)
        #Save MINE values!!!
        #Log to WnB here too
        

def train_MINE(GNN_model, GNN_dataset, MINE_models: MINE_networks, device):
    
    MINE_dataset = activations_dataset(GNN_model, GNN_dataset, MINE_models.config.activations_dataset_size)
    MINE_dataloader = DataLoader(MINE_dataset, batch_size = MINE_models.config.batch_size, shuffle = True, collate_fn = MINE_dataset.collate_fn)
    
    
    for layer in range(len(MINE_models)):
        MINE_models[layer]['input network'].train()
        MINE_models[layer]['label network'].train()
        
    for epoch in range(MINE_models.config.epochs):
        for batch_data in MINE_dataloader:
            batch_data.to(device)
            
            for layer in range(len(MINE_models)):
                #train the input network
                MINE_models[layer]['input network'].optimizer.zero_grad()
                output = MINE_models[layer]['input network'](batch_data)
                loss = None #Put the loss function in the MINE_models class.
                loss.backward()
                MINE_models[layer]['input network'].optimizer.step()
                
                #train the label network
                MINE_models[layer]['label network'].optimizer.zero_grad()
                output = MINE_models[layer]['label network'](batch_data)
                loss = None
                loss.backward()
                MINE_models[layer]['label network'].optimizer.step()