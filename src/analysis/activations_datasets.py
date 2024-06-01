from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class activations_dataset(Dataset):
    def __init__(self, GNN_model, GNN_dataset, dataset_size: int):
        GNN_model.eval()
        GNN_iterable_dataloader = iter(DataLoader(GNN_dataset, batch_size=GNN_model.config.batch_size, shuffle=True))
        current_dataset_size = 0
        
        self.dataset_size = dataset_size
        self.inputs = []
        self.hidden_activations = []
        self.labels = []
        
        while current_dataset_size < dataset_size:
            try:
                batch_data = next(GNN_iterable_dataloader)
            except StopIteration:
                GNN_iterable_dataloader = iter(DataLoader(GNN_dataset, batch_size=GNN_model.config.batch_size, shuffle=True))
                batch_data = next(GNN_iterable_dataloader)
            _, cached_activations = GNN_model.run_with_cache(batch_data.x)
            self.inputs.append(batch_data.x)
            self.hidden_activations.append(cached_activations)
            self.labels.append(batch_data.y)
            
            #Output is a list of torch tensors, without gradients.
            #Need to check how many nodes of data there are. Currently assuming there's one node...
            
            current_dataset_szie += 1
            
        self.inputs = self.inputs[:dataset_size]
        self.hidden_activations = self.hidden_activations[:dataset_size]
        self.labels = self.labels[:dataset_size]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return {'input': self.input_activaitons[index], 'hidden activations': self.hidden_activations[index], 'label': self.label_values[index]}