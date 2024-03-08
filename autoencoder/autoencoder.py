import torch
from torch import nn
from torch.utils.data import DataLoader
from representation.rep_utils import *
from representation.rep_class import *
from metamaterial_dataset import MetamaterialDataset
from representation.metamaterial_generation import random_metamaterial, plot_metamaterial


# Defines the autoencoder model
class MetamaterialAutoencoder(nn.Module):
    """
    A class for handling computations involving
    the metamaterial autoencoder.
    """

    # Initializes the structure of the NN
    def __init__(self):
        super().__init__()

        # Node position sizes
        node_layer_size = 300
        self.node_latent_size = 300

        # Node position encoder
        self.node_pos_encoder_stack = nn.Sequential(
            nn.Linear(NODE_POS_SIZE, node_layer_size),
            nn.ReLU(),
            nn.Linear(node_layer_size, node_layer_size),
            nn.ReLU(),
            nn.Linear(node_layer_size, self.node_latent_size),
        )

        # Node position decoder
        self.node_pos_decoder_stack = nn.Sequential(
            nn.Linear(self.node_latent_size, node_layer_size),
            nn.ReLU(),
            nn.Linear(node_layer_size, node_layer_size),
            nn.ReLU(),
            nn.Linear(node_layer_size, NODE_POS_SIZE),
            nn.Sigmoid()
        )

        # Edge adjacency sizes
        edge_layer_size = 500
        self.edge_latent_size = 500

        # Edge adjacency encoder
        self.edge_adj_encoder_stack = nn.Sequential(
            nn.Linear(EDGE_ADJ_SIZE, edge_layer_size),
            nn.ReLU(),
            nn.Linear(edge_layer_size, edge_layer_size),
            nn.ReLU(),
            nn.Linear(edge_layer_size, self.edge_latent_size),
        )

        # Edge Adjacency decoder
        self.edge_adj_decoder_stack = nn.Sequential(
            nn.Linear(self.edge_latent_size, edge_layer_size),
            nn.ReLU(),
            nn.Linear(edge_layer_size, edge_layer_size),
            nn.ReLU(),
            nn.Linear(edge_layer_size, EDGE_ADJ_SIZE),
            nn.Sigmoid()
        )

        # # Face adjacency encoder
        # self.face_adj_encoder_stack = nn.Sequential(
        #     nn.Linear(FACE_ADJ_SIZE, layer_size),
        #     nn.ReLU(),
        #     nn.Linear(layer_size, layer_size),
        #     nn.ReLU(),
        #     nn.Linear(layer_size, self.latent_size),
        # )

        # # Face adjacency decoder
        # self.face_adj_decoder_stack = nn.Sequential(
        #     nn.Linear(self.latent_size, layer_size),
        #     nn.ReLU(),
        #     nn.Linear(layer_size, layer_size),
        #     nn.ReLU(),
        #     nn.Linear(layer_size, FACE_ADJ_SIZE),
        #     nn.Sigmoid()
        # )

    
    def encoder(self, x):
        """
        Runs the network's encoder on the input.
        """

        # Prepares the individual representation arrays
        node_pos = x[:,:NODE_POS_SIZE]
        edge_adj = x[:,NODE_POS_SIZE:NODE_POS_SIZE+EDGE_ADJ_SIZE]
        # face_adj = x[:,NODE_POS_SIZE+EDGE_ADJ_SIZE:]

        # Runs the encoders
        return torch.cat((
            self.node_pos_encoder_stack(node_pos),
            self.edge_adj_encoder_stack(edge_adj),
            # self.face_adj_encoder_stack(face_adj)
        ), dim=1)

    
    def decoder(self, x):
        """
        Runs the network's decoder on the input.
        """

        # Prepares the individual representation arrays
        node_pos = x[:,:self.node_latent_size]
        edge_adj = x[:,self.node_latent_size:self.node_latent_size+self.edge_latent_size]
        # face_adj = x[:,self.latent_size*2:]

        # Runs the encoders
        return torch.cat((
            self.node_pos_decoder_stack(node_pos),
            self.edge_adj_decoder_stack(edge_adj),
            # self.face_adj_decoder_stack(face_adj)
        ), dim=1)


    # Defines a forward pass through the network
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
# Sets up the model and its traning methods
model = MetamaterialAutoencoder()
optim = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

# Sets up the dataloaders
train_data = DataLoader(MetamaterialDataset(150_000), batch_size=64, shuffle=True)
test_data = DataLoader(MetamaterialDataset(10_000), batch_size=64)
print("Data generated!")

def train():
    """
    Trains the autoencoder with the training data.
    """
    size = len(train_data.dataset)
    model.train()

    # Runs through each batch
    for batch, (X,y), in enumerate(train_data):

        # Prepares for backpropagation
        decoding = model(X)
        loss = loss_fn(decoding, y)

        # Backpropagation
        loss.backward()
        optim.step()
        optim.zero_grad()

        # Prints the loss every 200 batches
        if batch % 200 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"Loss: {loss:>7f} [{current}/{size}]")
    print()

def test():
    """
    Tests the autoencoder with test data
    """
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in test_data:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    
    # Prints the results of testing
    test_loss /= len(test_data)
    print(f"Test Avg Loss: {test_loss:>8f} \n")

    return test_loss

def load_model(filepath):
    """
    Loads the model at the given file, setting it to evaluation mode.

    filepath: str
        The path to the file containing the model state dictionary.

    Returns: MetamaterialAutoencoder
        The autoencoder network stored at the given file.
    """
    model = MetamaterialAutoencoder()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

def interpolate(material1, material2, interps, path, validate=False):
    """
    Linearly interpolates between the two given materials.

    material1: Metamaterial
        The base material for interpolation.

    material2: Metamaterial
        The material to be interpolated into.

    interps: int
        The number of interpolations to compute. Includes the starting 
        and ending metamaterial.

    path: str
        The path at which the intermediate metamaterials will be placed.

    validate: bool
        Whether to validate the interpolated metamaterials and remove
        any invalid edges/faces.
    """

    # Computes the latent representation of the two metamaterials
    m1_latent = model.encoder(material1.flatten_rep().reshape((1, NODE_POS_SIZE+EDGE_ADJ_SIZE+FACE_ADJ_SIZE)))
    m2_latent = model.encoder(material2.flatten_rep().reshape((1, NODE_POS_SIZE+EDGE_ADJ_SIZE+FACE_ADJ_SIZE)))

    # Runs through each interpolation
    for ind, alpha in enumerate([x/interps for x in range(interps+1)]):

        # Decodes the interpolated latent representation
        decoding = model.decoder(m1_latent*(1-alpha) + m2_latent*alpha)
        material = Metamaterial.from_tensor(decoding)

        # Validates the decoded representation
        if validate:
            material.remove_invalid_faces() # Removes faces without all edges in the rep
            material.remove_invalid_edges() # Removes edges intersecting with faces
            material.remove_invalid_faces() # Removes faces without all edges in the rep after edge removal

        plot_metamaterial(f"{path}/metamaterial{ind}.png", material, animate=False)

# Trains the model
epochs = 200
test_losses = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train()
    test_losses.append(test())
    torch.save(model.state_dict(), f"local_test/autoencoders/big_epoch{t}.pth")
print("Done!")

# Plots the test losses
import matplotlib.pyplot as plt
plt.figure()
plt.plot([x for x in range(epochs)], test_losses)
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.savefig("local_test/test_losses.png")
print("Plotted Test Losses!")
            
# Loads the model & interpolates
model = load_model("local_test/autoencoders/big_epoch180.pth")
interpolate(random_metamaterial(edge_prob=0.4, with_faces=False), random_metamaterial(edge_prob=0.4, with_faces=False), 50, "local_test/interpolated_trusses")

# # Visualize how good the model is doing
# material1 = random_metamaterial()
# decoding = model(material1.flatten_rep().reshape((1, NODE_POS_SIZE+EDGE_ADJ_SIZE+FACE_ADJ_SIZE)))[0,:].detach().numpy()
# decoding[NODE_POS_SIZE:] = (decoding[NODE_POS_SIZE:] > 0.5).astype(float)
# print(material1.flatten_rep().numpy())
# print(decoding)
# print(np.abs(material1.flatten_rep().numpy()-decoding))