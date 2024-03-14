import torch
from torch.utils.data import Dataset
from rotation_rep.representation.generation import *

class MetamaterialDataset(Dataset):

    # Initializes the dataset
    def __init__(self, class_size):
        super().__init__()
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Generates the random metamaterial data
        self.metamaterials = []
        for edge_prob in probs:
            self.metamaterials += [
                random_metamaterial(edge_prob=edge_prob).sort_rep().flatten_rep()
                    for _ in range(class_size*4//5)
            ]
        for edge_prob in probs:
            self.metamaterials += [
                random_metamaterial(edge_prob=edge_prob, grid_spacing=2).sort_rep().flatten_rep()
                    for _ in range(class_size//5)
            ]
        

    # The length of the dataset
    def __len__(self):
        return len(self.metamaterials)
    
    # Gets an input/label pair from the dataset
    def __getitem__(self, index):
        return self.metamaterials[index], self.metamaterials[index]
    