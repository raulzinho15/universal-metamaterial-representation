import torch
from torch.utils.data import Dataset

class MetamaterialVolumeDataset(Dataset):

    # Initializes the dataset
    def __init__(self, materials_file: str, volumes_file: str, device: str):
        super().__init__()
        self.metamaterials: torch.Tensor = torch.load(materials_file).to(device)
        self.volumes: torch.Tensor = torch.load(volumes_file).to(device)

    # The length of the dataset
    def __len__(self):
        return self.metamaterials.shape[0]
    
    # Gets an input/label pair from the dataset
    def __getitem__(self, index):
        return self.metamaterials[index], self.volumes[index]
    