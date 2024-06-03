import torch
from torch import nn

from representation.rep_utils import *
from representation.rep_class import *


# Defines the autoencoder model
class MetamaterialAE(nn.Module):
    """
    A class for handling computations involving
    the metamaterial autoencoder.
    """

    # Initializes the structure of the NN
    def __init__(self):
        super().__init__()

        # Sizes
        self.input_size = NODE_POS_SIZE + EDGE_ADJ_SIZE + EDGE_PARAMS_SIZE + FACE_ADJ_SIZE + FACE_PARAMS_SIZE
        self.hidden_size = self.input_size*2
        self.latent_size = self.input_size*2

        # Encoder
        self.encoder_stack = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.latent_size),
        )

        # Decoder
        self.decoder_stack = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.input_size),
        )


    def encoder(self, x):
        """
        Runs the network's encoder on the input.
        """
        return self.encoder_stack(x)
    

    def decoder(self, z):
        """
        Runs the network's decoder on the input.
        """
        return self.decoder_stack(z)


    # Defines a forward pass through the network
    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(model: MetamaterialAE, train_data, loss_fn, optim, verbose=True):
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
