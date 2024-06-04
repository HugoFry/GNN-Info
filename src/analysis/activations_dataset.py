from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class activations_dataset(Dataset):
    def __init__(self, GNN_model, GNN_dataset, dataset_size: int):
        GNN_model.eval()
        GNN_iterable_dataloader = iter(DataLoader(GNN_dataset, batch_size=GNN_model.config.batch_size, shuffle=True))
        current_dataset_size = 0
        
        self.dataset_size = dataset_size
        self.data = []
        
        while current_dataset_size < dataset_size:
            try:
                batch_data = next(GNN_iterable_dataloader)
            except StopIteration:
                GNN_iterable_dataloader = iter(DataLoader(GNN_dataset, batch_size=GNN_model.config.batch_size, shuffle=True))
                batch_data = next(GNN_iterable_dataloader)
            
            _, cached_activations = GNN_model.run_with_cache(batch_data.x)
            
            unbatched_data, batch_length = self.process_data(batch_data.x, cached_activations, batch_data.y)
            
            self.data += unbatched_data
            
            current_dataset_size += batch_length
            
        self.data = self.data[:dataset_size]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        """
        Returns a dictionary:
            Keys: integers indexing the layer.
                Index 0 is the input. 
                Index 1 is the 1st layer.
                Index 2 is the 2nd layer...
                Index -1 is the label. 
            Values: torch tensors of the GNN model activations.
        """
        return self.data[index]
    
    def process_data(self, input_batch, list_of_hidden_batch, label_batch):
        split_input = torch.split(input_batch, 1, dim = 0)
        split_input = [input.squeeze(0) for input in split_input]
        
        batch_length = len(split_input)
        
        split_label = torch.split(label_batch, 1, dim = 0)
        split_label = [label.squeeze(0) for label in split_label]
        
        list_of_split_hidden = [torch.split(hidden_batch, 1, dim = 0) for hidden_batch in list_of_hidden_batch]
        list_of_split_hidden = [[hidden.squeeze(0) for hidden in split_hidden] for split_hidden in list_of_split_hidden]
        
        unbatched_data = [split_input] + list_of_split_hidden + [split_label]
        unbatched_data = [list(item) for item in zip(*unbatched_data)] #Convert data from layers x batch, to batch x layers
        unbatched_data = [{layer: data for layer, data in enumerate(unbatched_data[index])} for index in batch_length]
        
        return unbatched_data, batch_length
    