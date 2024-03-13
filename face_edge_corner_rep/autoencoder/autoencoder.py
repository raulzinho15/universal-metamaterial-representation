import torch
from torch import nn

from representation.rep_utils import *
from representation.rep_class import *


# Defines the autoencoder model
class MetamaterialAutoencoder(nn.Module):
    """
    A class for handling computations involving
    the metamaterial autoencoder.
    """

    # Initializes the structure of the NN
    def __init__(self):
        super().__init__()

        # Encoder
        input_size = NODE_POS_SIZE+EDGE_ADJ_SIZE+FACE_ADJ_SIZE
        hidden_layer_size = input_size*2
        self.latent_size = hidden_layer_size
        self.encoder_stack = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, self.latent_size),
        )

        # Decoder
        self.decoder_stack = nn.Sequential(
            nn.Linear(self.latent_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, input_size),
        )

        # # Node position sizes
        # node_layer_size = 100
        # self.node_latent_size = 100

        # # Node position encoder
        # self.node_pos_encoder_stack = nn.Sequential(
        #     nn.Linear(NODE_POS_SIZE, node_layer_size),
        #     nn.ReLU(),
        #     nn.Linear(node_layer_size, node_layer_size),
        #     nn.ReLU(),
        #     nn.Linear(node_layer_size, self.node_latent_size),
        # )

        # # Node position decoder
        # self.node_pos_decoder_stack = nn.Sequential(
        #     nn.Linear(self.node_latent_size, node_layer_size),
        #     nn.ReLU(),
        #     nn.Linear(node_layer_size, node_layer_size),
        #     nn.ReLU(),
        #     nn.Linear(node_layer_size, NODE_POS_SIZE),
        #     nn.Sigmoid()
        # )

        # # Edge adjacency sizes
        # edge_layer_size = 400
        # self.edge_latent_size = 400

        # # Edge adjacency encoder
        # self.edge_adj_encoder_stack = nn.Sequential(
        #     nn.Linear(EDGE_ADJ_SIZE, edge_layer_size),
        #     nn.ReLU(),
        #     nn.Linear(edge_layer_size, edge_layer_size),
        #     nn.ReLU(),
        #     nn.Linear(edge_layer_size, self.edge_latent_size),
        # )

        # # Edge adjacency decoder
        # self.edge_adj_decoder_stack = nn.Sequential(
        #     nn.Linear(self.edge_latent_size, edge_layer_size),
        #     nn.ReLU(),
        #     nn.Linear(edge_layer_size, edge_layer_size),
        #     nn.ReLU(),
        #     nn.Linear(edge_layer_size, EDGE_ADJ_SIZE),
        #     nn.Sigmoid()
        # )

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
        return self.encoder_stack(x)

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
        return self.decoder_stack(x)

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

def train(model, train_data, loss_fn, optim):
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
    model = MetamaterialAutoencoder()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model
