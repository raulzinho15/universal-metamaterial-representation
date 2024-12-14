import torch
from torch import nn
from torch.utils.data import DataLoader
from representation.rep_utils import *


class MetamaterialAE(nn.Module):
    """
    A class for handling computations involving
    the metamaterial autoencoder.
    """

    def __init__(self, is_variational=False):
        """
        Constructs a metamaterial autoencoder with the given properties.

        is_variational: bool
            Whether the autoencoder is a variational autoencoder or not.
        """
        super().__init__()

        # Network sizes
        self.input_size = REP_SIZE
        self.hidden_size = self.input_size*4
        self.latent_size = self.input_size

        # Encoder
        self.encoder_stack = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.latent_size),
        )

        # Decoder
        self.decoder_stack = nn.Sequential(
            nn.Linear(in_features=self.latent_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.input_size),
        )


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the network's encoder on the input.
        """
        return self.encoder_stack(x)
    

    def sample_latent_space(self, latent_vector: torch.Tensor):
        """
        Assuming the autoencoder is a VAE, splits the latent
        vector into a normal distribution's mean/std and samples
        from the distribution.
        """

        # Finds the halfway mark
        halfway = latent_vector.shape[-1] // 2

        # Finds the distribution properties
        mean = latent_vector[..., :halfway]
        std = latent_vector[..., halfway:]

        # Samples the distribution
        eps = torch.randn(mean.shape)
        return mean + eps * std
    

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Runs the network's decoder on the input.
        """
        return self.decoder_stack(z)


    def forward(self, x: torch.Tensor):
        """
        Defines a forward pass through the network.
        """
        return self.decode(self.encode(x))


def train_epoch(model: MetamaterialAE, dataloader: DataLoader, loss_fn, optim, verbose=True, report_frequency=200) -> float:
    """
    Runs a training epoch on the model with the given data.

    model: `MetamaterialAE`
        The model to be used.

    dataloader: `DataLoader`
        The training data to use.

    loss_fn:
        The loss function to use.

    optim:
        The optimizer to use in training.

    verbose: `bool`, optional
        Whether batch progress will output to the terminal.

    report_frequency: `int`, optional
        The number of batches between which reports will be printed if
        the verbose option is chosen.

    Returns: `float`
        The loss from the epoch.
    """

    # Computes the size of the dataset
    dataset_size = len(dataloader.dataset)

    # Prepares values for the epoch
    samples_used = 0
    total_loss = 0

    # Sets up the model's mode
    model.train()

    # Runs through each batch
    for batch, (X,y), in enumerate(dataloader):

        # Computes the forward pass
        decoding = model(X)

        # Computes the loss
        loss: torch.Tensor = loss_fn(decoding, y)
        total_loss += loss.item() * X.shape[0]

        # Runs backpropagation and gradient descent
        loss.backward()
        optim.step()
        optim.zero_grad()

        # Prints the loss when the report frequency is met
        samples_used += X.shape[0]
        if verbose and (batch+1) % report_frequency == 0:
            print(f"Loss: {(total_loss / samples_used):>7f} [{samples_used}/{dataset_size}]")
    
    return total_loss


def test_epoch(model: MetamaterialAE, dataloader: DataLoader, loss_fn, verbose=False, report_frequency=200) -> tuple[float]:
    """
    Runs a testing epoch on the model with the given data.

    model: `MetamaterialAE`
        The model to be used.

    dataloader: `DataLoader`
        The testing data to use.

    loss_fn:
        The loss function to use.

    verbose: `bool`, optional
        Whether batch progress will output to the terminal.

    report_frequency: `int`, optional
        The number of batches between which reports will be printed if
        the verbose option is chosen.

    Returns: `tuple[float]`
        1) The loss from the epoch.
        2) The average absolute node position error.
        3) The fraction of edges correctly decoded.
        4) The average absolute edge parameters error.
        5) The fraction of faces correctly decoded.
        6) The average absolute face parameters error.
    """

    # Computes the size of the dataset
    dataset_size = len(dataloader.dataset)

    # Prepares values for the epoch
    samples_used = 0
    total_loss = 0
    node_pos_error = 0
    correct_edges = 0
    edge_params_error = 0
    correct_faces = 0
    face_params_error = 0

    # Sets up the model's mode
    model.eval()

    # Runs through each batch
    for batch, (X,y), in enumerate(dataloader):

        # Computes the forward pass
        decoding = model(X)

        # Computes the loss
        loss: torch.Tensor = loss_fn(decoding, y)
        total_loss += loss.item() * X.shape[0]

        # Computes the average absolute error in node positions
        y_nodes = y[:,:NODE_POS_SIZE]
        decoding_nodes = decoding[:,:NODE_POS_SIZE]
        node_pos_error += torch.sum(torch.abs(y_nodes-decoding_nodes)).item() / NODE_POS_SIZE

        # Computes the proportion of edges that were decoded correctly
        y_edges = y[:,NODE_POS_SIZE:][:,:EDGE_ADJ_SIZE]
        decoding_edges = decoding[:,NODE_POS_SIZE:][:,:EDGE_ADJ_SIZE]
        correct_edges += torch.sum(torch.abs(decoding_edges-y_edges) < 0.5).item() / EDGE_ADJ_SIZE

        # Computes the average absolute error in edge parameters
        y_edge_params = y[:,NODE_POS_SIZE+EDGE_ADJ_SIZE:][:,:EDGE_PARAMS_SIZE]
        decoding_edge_params = decoding[:,NODE_POS_SIZE+EDGE_ADJ_SIZE:][:,:EDGE_PARAMS_SIZE]
        edge_params_error += torch.sum(torch.abs(y_edge_params-decoding_edge_params)).item() / EDGE_PARAMS_SIZE

        # Computes the proportion of faces that were decoded correctly
        y_faces = y[:,NODE_POS_SIZE+EDGE_ADJ_SIZE+EDGE_PARAMS_SIZE:][:,:FACE_ADJ_SIZE]
        decoding_faces = decoding[:,NODE_POS_SIZE+EDGE_ADJ_SIZE+EDGE_PARAMS_SIZE:][:,:FACE_ADJ_SIZE]
        correct_faces += torch.sum(torch.abs(decoding_faces-y_faces) < 0.5).item() / FACE_ADJ_SIZE

        # Computes the average absolute error in face parameters
        y_face_params = y[:,NODE_POS_SIZE+EDGE_ADJ_SIZE+EDGE_PARAMS_SIZE+FACE_ADJ_SIZE:][:,:FACE_PARAMS_SIZE]
        decoding_face_params = decoding[:,NODE_POS_SIZE+EDGE_ADJ_SIZE+EDGE_PARAMS_SIZE+FACE_ADJ_SIZE:][:,:FACE_PARAMS_SIZE]
        face_params_error += torch.sum(torch.abs(y_face_params-decoding_face_params)).item() / FACE_PARAMS_SIZE

        # Prints the loss when the report frequency is met
        samples_used += X.shape[0]
        if verbose and (batch+1) % report_frequency == 0:
            print(f"Loss: {(total_loss / samples_used):>7f} [{samples_used}/{dataset_size}]")
    
    # Averages the output values
    total_loss /= samples_used
    node_pos_error /= samples_used
    correct_edges /= samples_used
    edge_params_error /= samples_used
    correct_faces /= samples_used
    face_params_error /= samples_used

    return total_loss, node_pos_error, correct_edges, edge_params_error, correct_faces, face_params_error


def load_model(filepath):
    """
    Loads the model at the given file, setting it to evaluation mode.

    filepath: str
        The path to the file containing the model state dictionary.

    Returns: MetamaterialAutoencoder
        The autoencoder network stored at the given file.
    """

    # Loads the model
    model = MetamaterialAE()
    model.load_state_dict(torch.load(filepath))

    # Sets to evaluation mode
    model.eval()

    return model

