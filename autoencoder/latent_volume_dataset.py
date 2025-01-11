import torch
from torch.utils.data import Dataset

class LatentVolumeDataset(Dataset):

    # Initializes the dataset
    def __init__(self, latents_file: str, volumes_file: str, device: str):
        super().__init__()
        self.latents: torch.Tensor = torch.load(latents_file).to(device)
        self.volumes: torch.Tensor = torch.load(volumes_file).to(device)

    # The length of the dataset
    def __len__(self):
        return self.latents.shape[0]
    
    # Gets an input/label pair from the dataset
    def __getitem__(self, index):
        return self.latents[index], self.volumes[index]
    