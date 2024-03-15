import torch
from torch import nn

from rotation_rep.representation.rep_utils import *
from rotation_rep.representation.rep_class import *


# Defines the autoencoder model
class MetamaterialAE(nn.Module):
    """
    A class for handling computations involving
    the metamaterial variational autoencoder.
    """

    # Initializes the structure of the NN
    def __init__(self):
        super().__init__()

        # Node position sizes
        self.node_input_size = NODE_POS_SIZE
        self.node_hidden_size = NODE_POS_SIZE*2
        self.node_latent_size = NODE_POS_SIZE*2

        # Node position encoder
        self.node_encoder_stack = nn.Sequential(
            nn.Linear(self.node_input_size, self.node_hidden_size),
            nn.ReLU(),
            nn.Linear(self.node_hidden_size, self.node_hidden_size),
            nn.ReLU(),
            nn.Linear(self.node_hidden_size, self.node_latent_size),
        )

        # Node position decoder
        self.node_decoder_stack = nn.Sequential(
            nn.Linear(self.node_latent_size, self.node_hidden_size),
            nn.ReLU(),
            nn.Linear(self.node_hidden_size, self.node_hidden_size),
            nn.ReLU(),
            nn.Linear(self.node_hidden_size, self.node_input_size),
        )

        # Edge adjacency sizes
        self.edge_input_size = EDGE_ADJ_SIZE
        self.edge_hidden_size = EDGE_ADJ_SIZE*2
        self.edge_latent_size = EDGE_ADJ_SIZE*2

        # Edge adjacency encoder
        self.edge_encoder_stack = nn.Sequential(
            nn.Linear(self.edge_input_size, self.edge_hidden_size),
            nn.ReLU(),
            nn.Linear(self.edge_hidden_size, self.edge_hidden_size),
            nn.ReLU(),
            nn.Linear(self.edge_hidden_size, self.edge_latent_size),
        )

        # Edge adjacency decoder
        self.edge_decoder_stack = nn.Sequential(
            nn.Linear(self.edge_latent_size, self.edge_hidden_size),
            nn.ReLU(),
            nn.Linear(self.edge_hidden_size, self.edge_hidden_size),
            nn.ReLU(),
            nn.Linear(self.edge_hidden_size, self.edge_input_size),
        )

        # Face adjacency sizes
        self.face_input_size = FACE_ADJ_SIZE
        self.face_hidden_size = FACE_ADJ_SIZE*2
        self.face_latent_size = FACE_ADJ_SIZE*2

        # Face adjacency encoder
        self.face_encoder_stack = nn.Sequential(
            nn.Linear(self.face_input_size, self.face_hidden_size),
            nn.ReLU(),
            nn.Linear(self.face_hidden_size, self.face_hidden_size),
            nn.ReLU(),
            nn.Linear(self.face_hidden_size, self.face_latent_size),
        )

        # Face adjacency decoder
        self.face_decoder_stack = nn.Sequential(
            nn.Linear(self.face_latent_size, self.face_hidden_size),
            nn.ReLU(),
            nn.Linear(self.face_hidden_size, self.face_hidden_size),
            nn.ReLU(),
            nn.Linear(self.face_hidden_size, self.face_input_size),
        )


    def encoder(self, x):
        """
        Runs the network's encoder on the input.
        """

        # Separates the components
        node_pos = x[:,:self.node_input_size]
        edge_adj = x[:,self.node_input_size:self.node_input_size+self.edge_input_size]
        face_adj = x[:,self.node_input_size+self.edge_input_size:]

        # Passes through the encoding layers
        node_encoding = self.node_encoder_stack(node_pos)
        edge_encoding = self.edge_encoder_stack(edge_adj)
        face_encoding = self.face_encoder_stack(face_adj)

        return torch.cat((node_encoding, edge_encoding, face_encoding), dim=1)
    

    def decoder(self, z):
        """
        Runs the network's decoder on the input.
        """

        # Separates the components
        node_pos = z[:,:self.node_latent_size]
        edge_adj = z[:,self.node_latent_size:self.node_latent_size+self.edge_latent_size]
        face_adj = z[:,self.node_latent_size+self.edge_latent_size:]

        # Passes through the decoding layers
        node_decoding = self.node_decoder_stack(node_pos)
        edge_decoding = self.edge_decoder_stack(edge_adj)
        face_decoding = self.face_decoder_stack(face_adj)

        return torch.cat((node_decoding, edge_decoding, face_decoding), dim=1)


    # Defines a forward pass through the network
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train(model, train_data, loss_fn, optim, verbose=True):
    """
    Trains the model with the given training data.

    model: MetamaterialAutoencoder
        The model to be trained.

    train_data: DataLoader
        The data from a MetamaterialDataset with which to train.

    loss_fn:
        The loss function to use in training.

    optim:
        The optimizer to use in training.

    verbose: bool
        Whether batch progress will output to the terminal.
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
        if verbose and batch % 200 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"Loss: {loss:>7f} [{current}/{size}]")
    print()

def test(model, test_data, loss_fn):
    """
    Tests the autoencoder with the given test data.

    model: MetamaterialAutoencoder
        The model to be tested.

    test_data: DataLoader
        The data from a MetamaterialDataset with which to test.

    loss_fn:
        The loss function to use in testing.
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
    model = MetamaterialAE()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model
