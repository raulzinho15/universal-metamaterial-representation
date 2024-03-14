from torch.utils.data import Dataset
from rotation_rep.representation.generation import *

class MetamaterialDataset(Dataset):

    # Initializes the dataset
    def __init__(self, class_size):
        super().__init__()
        probs = [x/20 for x in range(1, 20)]

        # Generates the random metamaterial data
        per_class = class_size//(19*19)
        self.metamaterials = []
        for edge_prob in probs:
            for face_prob in probs:
                self.metamaterials += [
                    random_metamaterial(edge_prob=edge_prob, face_prob=face_prob).sort_rep().flatten_rep()
                        for _ in range(per_class*4//5)
                ]
                self.metamaterials += [
                    random_metamaterial(edge_prob=edge_prob, face_prob=face_prob, grid_spacing=2).sort_rep().flatten_rep()
                        for _ in range(per_class//5)
                ]
            print(f"Generated metamaterials with edge probability: {edge_prob}")
        

    # The length of the dataset
    def __len__(self):
        return len(self.metamaterials)
    
    # Gets an input/label pair from the dataset
    def __getitem__(self, index):
        return self.metamaterials[index], self.metamaterials[index]
    