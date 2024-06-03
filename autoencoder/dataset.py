from torch.utils.data import Dataset
from representation.generation import *

class MetamaterialDataset(Dataset):

    # Initializes the dataset
    def __init__(self, class_size):
        super().__init__()
        probs = [x/10 for x in range(10+1)]

        # Generates the random metamaterial data
        per_class = class_size//(len(probs)**2)
        self.metamaterials = []
        for edge_prob in probs:
            for face_prob in probs:

                # Skips empty metamaterials
                if edge_prob == face_prob == 0.0:
                    continue

                self.metamaterials += [
                    random_metamaterial(edge_prob=edge_prob, face_prob=face_prob, connected=True).flatten_rep()
                        for _ in range(per_class*3//10)
                ]
                self.metamaterials += [
                    random_metamaterial(edge_prob=edge_prob, face_prob=face_prob, wavy_edges=True, connected=True).flatten_rep()
                        for _ in range(per_class*3//10)
                ]
                self.metamaterials += [
                    random_metamaterial(edge_prob=edge_prob, face_prob=face_prob, grid_spacing=2, connected=True).flatten_rep()
                        for _ in range(per_class*2//10)
                ]
                self.metamaterials += [
                    random_metamaterial(edge_prob=edge_prob, face_prob=face_prob, grid_spacing=2, wavy_edges=True, connected=True).flatten_rep()
                        for _ in range(per_class*2//10)
                ]
                print(f"Generated metamaterials with (edge,face) probability: ({edge_prob},{face_prob})")
        

    # The length of the dataset
    def __len__(self):
        return len(self.metamaterials)
    
    # Gets an input/label pair from the dataset
    def __getitem__(self, index):
        return self.metamaterials[index], self.metamaterials[index]
    