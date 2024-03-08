import torch
from torch.utils.data import Dataset
from representation.metamaterial_generation import *

class MetamaterialDataset(Dataset):

    # Initializes the dataset
    def __init__(self, size):
        super().__init__()

        # Generates the random metamaterial data
        self.metamaterials = [
            random_metamaterial().flatten_rep()[:NODE_POS_SIZE+EDGE_ADJ_SIZE]
                for _ in range(size)
        ]
        

    # The length of the dataset
    def __len__(self):
        return len(self.metamaterials)
    
    # Gets an input/label pair from the dataset
    def __getitem__(self, index):
        return self.metamaterials[index], self.metamaterials[index]
    