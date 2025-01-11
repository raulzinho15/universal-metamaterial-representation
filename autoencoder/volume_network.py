from torch import nn
from torch.utils.data import DataLoader
import torch

class VolumeNetwork(nn.Module):

    def __init__(self, input_size: int, device: str):

        # Stores the network sizes
        self.input_size = input_size
        self.hidden_size = input_size*4
        self.output_size = 1

        self.stack = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.output_size),
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)
    
def train_epoch(epoch: int, model: VolumeNetwork, dataloader: DataLoader, loss_fn, optim, verbose=True, report_frequency=200):

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
        pred: torch.Tensor = model(X)

        # Computes the loss
        loss: torch.Tensor = loss_fn(pred, y)

        # Computes the total loss
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

def test_epoch(model: VolumeNetwork, dataloader: DataLoader, loss_fn, verbose=False, report_frequency=200):

    # Computes the size of the dataset
    dataset_size = len(dataloader.dataset)

    # Prepares values for the epoch
    samples_used = 0
    total_loss = 0

    # Sets up the model's mode
    model.eval()

    # Runs through each batch
    for batch, (X,y), in enumerate(dataloader):

        # Computes the forward pass
        pred: torch.Tensor = model(X)

        # Computes the loss
        loss: torch.Tensor = loss_fn(pred, y)

        # Computes the total loss
        total_loss += loss.item() * X.shape[0]

        # Prints the loss when the report frequency is met
        samples_used += X.shape[0]
        if verbose and (batch+1) % report_frequency == 0:
            print(f"Test Loss: {(total_loss / samples_used):>7f} [{samples_used}/{dataset_size}]")

    return total_loss