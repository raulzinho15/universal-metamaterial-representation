import torch
import math
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

        # Node Position Encoder
        self.node_pos_input_size = NODE_POS_SIZE
        self.node_pos_hidden_size = self.node_pos_input_size*4
        self.node_pos_latent_size = self.node_pos_input_size//3
        self.node_pos_encoder = nn.Sequential(
            nn.Linear(in_features=self.node_pos_input_size, out_features=self.node_pos_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.node_pos_hidden_size, out_features=self.node_pos_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.node_pos_hidden_size, out_features=self.node_pos_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.node_pos_hidden_size, out_features=self.node_pos_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.node_pos_hidden_size, out_features=self.node_pos_latent_size),
        )

        # Edge Adjacency Encoder
        self.edge_adj_input_size = EDGE_ADJ_SIZE
        self.edge_adj_hidden_size = self.edge_adj_input_size*4
        self.edge_adj_latent_size = self.edge_adj_input_size//3
        self.edge_adj_encoder = nn.Sequential(
            nn.Linear(in_features=self.edge_adj_input_size, out_features=self.edge_adj_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.edge_adj_hidden_size, out_features=self.edge_adj_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.edge_adj_hidden_size, out_features=self.edge_adj_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.edge_adj_hidden_size, out_features=self.edge_adj_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.edge_adj_hidden_size, out_features=self.edge_adj_latent_size),
        )

        # Edge Parameters Encoder
        self.edge_params_input_size = EDGE_PARAMS_SIZE
        self.edge_params_hidden_size = self.edge_params_input_size*4
        self.edge_params_latent_size = self.edge_params_input_size//3
        self.edge_params_encoder = nn.Sequential(
            nn.Linear(in_features=self.edge_params_input_size, out_features=self.edge_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.edge_params_hidden_size, out_features=self.edge_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.edge_params_hidden_size, out_features=self.edge_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.edge_params_hidden_size, out_features=self.edge_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.edge_params_hidden_size, out_features=self.edge_params_latent_size),
        )

        # Face Adjacency Encoder
        self.face_adj_input_size = FACE_ADJ_SIZE
        self.face_adj_hidden_size = self.face_adj_input_size*4
        self.face_adj_latent_size = self.face_adj_input_size//3
        self.face_adj_encoder = nn.Sequential(
            nn.Linear(in_features=self.face_adj_input_size, out_features=self.face_adj_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.face_adj_hidden_size, out_features=self.face_adj_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.face_adj_hidden_size, out_features=self.face_adj_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.face_adj_hidden_size, out_features=self.face_adj_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.face_adj_hidden_size, out_features=self.face_adj_latent_size),
        )

        # Face Parameters Encoder
        self.face_params_input_size = FACE_PARAMS_SIZE
        self.face_params_hidden_size = self.face_params_input_size*4
        self.face_params_latent_size = self.face_params_input_size//3
        self.face_params_encoder = nn.Sequential(
            nn.Linear(in_features=self.face_params_input_size, out_features=self.face_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.face_params_hidden_size, out_features=self.face_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.face_params_hidden_size, out_features=self.face_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.face_params_hidden_size, out_features=self.face_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.face_params_hidden_size, out_features=self.face_params_latent_size),
        )

        # Global Parameters Encoder
        self.global_params_input_size = GLOBAL_PARAMS_SIZE
        self.global_params_hidden_size = self.global_params_input_size*4
        self.global_params_latent_size = math.ceil(self.global_params_input_size/3)
        self.global_params_encoder = nn.Sequential(
            nn.Linear(in_features=self.global_params_input_size, out_features=self.global_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.global_params_hidden_size, out_features=self.global_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.global_params_hidden_size, out_features=self.global_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.global_params_hidden_size, out_features=self.global_params_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.global_params_hidden_size, out_features=self.global_params_latent_size),
        )

        # Topology Encoder
        self.topology_input_size = EDGE_ADJ_SIZE+FACE_ADJ_SIZE
        self.topology_hidden_size = self.topology_input_size*4
        self.topology_latent_size = self.topology_input_size//3
        self.topology_encoder = nn.Sequential(
            nn.Linear(in_features=self.topology_input_size, out_features=self.topology_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.topology_hidden_size, out_features=self.topology_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.topology_hidden_size, out_features=self.topology_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.topology_hidden_size, out_features=self.topology_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.topology_hidden_size, out_features=self.topology_latent_size),
        )

        # Geometry Encoder
        self.geometry_input_size = NODE_POS_SIZE+EDGE_PARAMS_SIZE+FACE_PARAMS_SIZE+GLOBAL_PARAMS_SIZE
        self.geometry_hidden_size = self.geometry_input_size*4
        self.geometry_latent_size = self.geometry_input_size//3
        self.geometry_encoder = nn.Sequential(
            nn.Linear(in_features=self.geometry_input_size, out_features=self.geometry_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.geometry_hidden_size, out_features=self.geometry_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.geometry_hidden_size, out_features=self.geometry_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.geometry_hidden_size, out_features=self.geometry_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.geometry_hidden_size, out_features=self.geometry_latent_size),
        )

        # Node Position Decoder
        input_size = self.node_pos_latent_size+self.geometry_latent_size
        hidden_size = input_size*4
        output_size = self.node_pos_input_size
        self.node_pos_decoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )

        # Edge Adjacency Decoder
        input_size = self.edge_adj_latent_size+self.topology_latent_size
        hidden_size = input_size*4
        output_size = self.edge_adj_input_size
        self.edge_adj_decoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )

        # Edge Parameters Decoder
        input_size = self.edge_params_latent_size+self.geometry_latent_size
        hidden_size = input_size*4
        output_size = self.edge_params_input_size
        self.edge_params_decoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )

        # Face Adjacency Decoder
        input_size = self.face_adj_latent_size+self.topology_latent_size
        hidden_size = input_size*4
        output_size = self.face_adj_input_size
        self.face_adj_decoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )

        # Face Parameters Decoder
        input_size = self.face_params_latent_size+self.geometry_latent_size
        hidden_size = input_size*4
        output_size = self.face_params_input_size
        self.face_params_decoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )

        # Global Parameters Decoder
        input_size = self.global_params_latent_size+self.geometry_latent_size
        hidden_size = input_size*4
        output_size = self.global_params_input_size
        self.global_params_decoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the network's encoder on the input.
        """

        # Stores the indices
        edge_adj_index = NODE_POS_SIZE
        edge_params_index = edge_adj_index+EDGE_ADJ_SIZE
        face_adj_index = edge_params_index+EDGE_PARAMS_SIZE
        face_params_index = face_adj_index+FACE_ADJ_SIZE
        global_params_index = face_params_index+FACE_PARAMS_SIZE

        # Stores the values from the representation
        node_pos =      x[...,                     : edge_adj_index     ]
        edge_adj =      x[..., edge_adj_index      : edge_params_index  ]
        edge_params =   x[..., edge_params_index   : face_adj_index     ]
        face_adj =      x[..., face_adj_index      : face_params_index  ]
        face_params =   x[..., face_params_index   : global_params_index]
        global_params = x[..., global_params_index :                    ]

        # Stores the latent values
        node_pos_latent = self.node_pos_encoder(node_pos)
        edge_adj_latent = self.edge_adj_encoder(edge_adj)
        edge_params_latent = self.edge_params_encoder(edge_params)
        face_adj_latent = self.face_adj_encoder(face_adj)
        face_params_latent = self.face_params_encoder(face_params)
        global_params_latent = self.global_params_encoder(global_params)
        topology_latent = self.topology_encoder(torch.cat([edge_adj, face_adj], dim=-1))
        geometry_latent = self.geometry_encoder(torch.cat([node_pos, edge_params, face_params, global_params], dim=-1))

        return torch.cat([
            node_pos_latent,
            edge_adj_latent,
            edge_params_latent,
            face_adj_latent,
            face_params_latent,
            global_params_latent,
            topology_latent,
            geometry_latent,
        ], dim=-1)
    

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

        # Stores the indices
        edge_adj_index = self.node_pos_latent_size
        edge_params_index = edge_adj_index + self.edge_adj_latent_size
        face_adj_index = edge_params_index + self.edge_params_latent_size
        face_params_index = face_adj_index + self.face_adj_latent_size
        global_params_index = face_params_index + self.face_params_latent_size
        topology_index = global_params_index + self.global_params_latent_size
        geometry_index = topology_index + self.topology_latent_size

        # Stores the values from the representation
        node_pos_latent =      z[...,                     : edge_adj_index     ]
        edge_adj_latent =      z[..., edge_adj_index      : edge_params_index  ]
        edge_params_latent =   z[..., edge_params_index   : face_adj_index     ]
        face_adj_latent =      z[..., face_adj_index      : face_params_index  ]
        face_params_latent =   z[..., face_params_index   : global_params_index]
        global_params_latent = z[..., global_params_index : topology_index     ]
        topology_latent =      z[..., topology_index      : geometry_index     ]
        geometry_latent =      z[..., geometry_index      :                    ]

        # Stores the decoded values
        node_pos = self.node_pos_decoder(torch.cat([node_pos_latent, geometry_latent], dim=-1))
        edge_adj = self.edge_adj_decoder(torch.cat([edge_adj_latent, topology_latent], dim=-1))
        edge_params = self.edge_params_decoder(torch.cat([edge_params_latent, geometry_latent], dim=-1))
        face_adj = self.face_adj_decoder(torch.cat([face_adj_latent, topology_latent], dim=-1))
        face_params = self.face_params_decoder(torch.cat([face_params_latent, geometry_latent], dim=-1))
        global_params = self.global_params_decoder(torch.cat([global_params_latent, geometry_latent], dim=-1))

        return torch.cat([
            node_pos,
            edge_adj,
            edge_params,
            face_adj,
            face_params,
            global_params,
        ], dim=-1)


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

