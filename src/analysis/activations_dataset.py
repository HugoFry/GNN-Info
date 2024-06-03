from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class activations_dataset(Dataset):
    def __init__(self, GNN_model, GNN_dataset, dataset_size: int):
        GNN_model.eval()
        GNN_iterable_dataloader = iter(DataLoader(GNN_dataset, batch_size=GNN_model.config.batch_size, shuffle=True))
        current_dataset_size = 0
        
        self.dataset_size = dataset_size
        self.inputs = []
        self.labels = []
        
        while current_dataset_size < dataset_size:
            try:
                batch_data = next(GNN_iterable_dataloader)
            except StopIteration:
                GNN_iterable_dataloader = iter(DataLoader(GNN_dataset, batch_size=GNN_model.config.batch_size, shuffle=True))
                batch_data = next(GNN_iterable_dataloader)
            _, cached_activations = GNN_model.run_with_cache(batch_data.x)
            
            #####
            #This won't worl as the x/y data has a different shape.
            #This also is rubsih - I need to split the tensors along the batch dimension.
            #Also put this code into a seperate method? some data processing method
            #####
            concatinaed_inputs = []
            concatinaed_labels = []
            for layer_activation in cached_activations:
                concatinaed_inputs.append(torch.cat((batch_data.x, layer_activation), dim = -1))
                concatinaed_labels.append(torch.cat((batch_data.y, layer_activation), dim = -1))
            
            
            current_dataset_szie += 1
            
            #Output is a list of torch tensors, without gradients.
            #Need to check how many nodes of data there are. Currently assuming there's one node... EG
            ######
            #THIS IS CURRENTLY BATCHED!!
            ######
            
        self.inputs = self.inputs[:dataset_size]
        self.hidden_activations = self.hidden_activations[:dataset_size]
        self.labels = self.labels[:dataset_size]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return {'input': self.inputs[index], 'label': self.labels[index]}
    
    def collate_fn(self, batch):
        # Initialize the batched data structure
        batched_data = {'input': [], 'label': []}
        for items in zip(*[sample['input'] for sample in batch]):
                batched_data['input'].append(torch.stack(items))
                
        for items in zip(*[sample['label'] for sample in batch]):
                batched_data['label'].append(torch.stack(items))
                
        return batched_data
    
"""
Need to change to get the following:
    Return type a dictionary with keys 'input' and 'label'.
    the values of the dictionary should be a list (indexing different layers) of torch tensors (represnting the concatination of two tensors)
"""