from typing import Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


#################### Added for the graph network ####################
class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, int(hidden_channels / 2))
        self.conv3 = GCNConv(int(hidden_channels / 2), out_channels)

    def forward(self, x, edge_index, batch):
        # Apply the first GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Apply the second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global pooling to obtain the graph embedding
        x = global_mean_pool(x, batch)

        return x


class RingAttractorNetworkGraph:
    """Graph representing the ring attractor network"""

    def __init__(self, nb_of_nodes: int) -> None:
        # Note that final center node is of idx nb_of_nodes-1

        self.nb_of_nodes = nb_of_nodes
        self.ran_graph = nx.MultiDiGraph()

        # Add nb_of_nodes nodes
        nodes = range(0, self.nb_of_nodes - 1)
        self.ran_graph.add_nodes_from(nodes)

        # Add bidirectional connections for the first nb_of_nodes-1 nodes
        for i in range(0, self.nb_of_nodes - 2):
            self.ran_graph.add_edge(i, i + 1)
            self.ran_graph.add_edge(i + 1, i)

        # Add connections between idx nb_of_nodes-2 and 0

        self.ran_graph.add_edge(self.nb_of_nodes - 2, 0)
        self.ran_graph.add_edge(0, self.nb_of_nodes - 2)

        # Add unidirectional connections for the last node
        for i in range(0, self.nb_of_nodes - 1):
            self.ran_graph.add_edge(self.nb_of_nodes - 1, i)

        # add edge features
        self.edge_feat = self.get_edge_features()
        for idx, (_, _, e) in enumerate(self.ran_graph.edges(data=True)):
            e["feature"] = self.edge_feat[idx]

    # Function to prepare the node features
    def get_node_features(
        self,
        image_encoding: torch.Tensor,
        nb_connections: int = 2,  # TODO: CHANGE TO OPTIONAL
        odometry: Optional[torch.Tensor] = None,
    ):
        """
        Build the node features tensor

        Args:
            nb_nodes (int): total nb of nodes (including the center node), odd number
            image_encoding (torch.Tensor): embeddings of the image
            node_pos_encoding (_type_): circular positional encoding
            nb_connections (Optional[int], optional): nb of connections of each node.
            Defaults to 2.
            odometry (Optional[torch.Tensor], optional): odom data Defaults to None.
        """
        positions = self.distribute_nodes_on_circle(radius=1)

        nb_connections_tensor = torch.tensor(nb_connections).unsqueeze(0)
        nb_connections_tensor = nb_connections_tensor.repeat(self.nb_of_nodes - 1, 1)
        nb_connections_tensor = torch.cat(
            (
                nb_connections_tensor,
                torch.tensor([self.nb_of_nodes - 1]).unsqueeze(0),
            ),
            0,
        )
        # print("before repeat", image_encoding.shape)
        image_encoding = image_encoding.repeat(self.nb_of_nodes, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_encoding = image_encoding.to(device)
        nb_connections_tensor = nb_connections_tensor.to(device)
        positions = positions.to(device)

        feature_vector = torch.concat(
            (image_encoding, nb_connections_tensor, positions), 1
        )
        # print("before repeat", image_encoding.shape)
        if odometry is not None:
            odometry = odometry.repeat(self.nb_of_nodes, 1)
            feature_vector = torch.concat((feature_vector, odometry), 1)

        return feature_vector

    # Function to prepare the edge features
    def get_edge_features(self):
        """
        Build the edge features tensor

        Args:
            nb_nodes (int): total nb of nodes (including the center node), odd number
        """
        # Define the edge features tensor with shape (2 * nb_nodes + nb_nodes, 1)
        # first 2 * nb_nodes are for the connections between the nodes, should be 1
        # last nb_nodes are for the connections between the new node and the existing
        # nodes, should be -1
        edge_features = torch.ones(
            (2 * (self.nb_of_nodes - 1) + (self.nb_of_nodes - 1), 1)
        )
        edge_features[2 * (self.nb_of_nodes - 1) :, :] = -1

        return edge_features

    def get_edge_index(self):
        # create edge index from
        adj = nx.to_scipy_sparse_array(self.ran_graph).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        return edge_index

    def distribute_nodes_on_circle(self, radius=1):
        angles = np.linspace(
            0, 2 * np.pi, self.nb_of_nodes - 1, endpoint=False
        )  # Divide the circle into equal angles
        x = radius * np.cos(angles)  # Calculate x-coordinates using cosine function
        y = radius * np.sin(angles)  # Calculate y-coordinates using sine function
        positions = np.column_stack(
            (x, y)
        )  # Stack x and y coordinates as column vectors

        positions = np.vstack((positions, np.array([0, 0])))
        return torch.tensor(positions)
