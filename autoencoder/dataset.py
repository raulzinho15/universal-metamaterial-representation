import torch
from torch.utils.data import Dataset

class MetamaterialDataset(Dataset):

    # Initializes the dataset
    def __init__(self, dataset_file: str, device):
        super().__init__()
        self.metamaterials: torch.Tensor = torch.load(dataset_file).to(device)    

    # The length of the dataset
    def __len__(self):
        return len(self.metamaterials)
    
    # Gets an input/label pair from the dataset
    def __getitem__(self, index):
        return self.metamaterials[index], self.metamaterials[index]
    