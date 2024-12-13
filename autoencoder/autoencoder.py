import torch
from torch import nn
from torch.utils.data import DataLoader
from representation.rep_utils import *
from representation.rep_class import *


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


def run_epoch(model: MetamaterialAE, dataloader: DataLoader, loss_fn, optim: None | torch.optim.Adam = None, train=True, verbose=True, report_frequency=200) -> float:
    """
    Runs an epoch on the model with the given data.

    model: MetamaterialAutoencoder
        The model to be used.

    dataloader: DataLoader
        The data from a MetamaterialDataset to use.

    loss_fn:
        The loss function to use.

    optim:
        The optimizer to use in training.

    train: bool
        Whether the model will be trained or not.

    verbose: bool
        Whether batch progress will output to the terminal.

    report_frequency: int
        The number of batches between which reports will be printed if
        the verbose option is chosen.

    Returns: float
        The loss from the epoch.
    """

    # Computes the size of the dataset
    dataset_size = len(dataloader.dataset)

    # Prepares values for the epoch
    samples_used = 0
    total_loss = 0

    # Sets up the model's mode
    if train:
        model.train()
    else:
        model.eval()

    # Runs through each batch
    for batch, (X,y), in enumerate(dataloader):

        # Computes the forward pass
        decoding = model(X)

        # Computes the loss
        loss: torch.Tensor = loss_fn(decoding, y)
        total_loss += loss.item()

        # Runs backpropagation and gradient descent
        if train:
            loss.backward()
            optim.step()
            optim.zero_grad()

        # Prints the loss when the report frequency is met
        samples_used += X.shape[0]
        if verbose and batch % report_frequency == 0:
            print(f"Loss: {(total_loss / samples_used):>7f} [{samples_used}/{dataset_size}]")
    
    return total_loss


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

