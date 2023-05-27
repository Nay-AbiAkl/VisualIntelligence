#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

import torch
from gym import spaces
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.utils.common import CategoricalNet, GaussianNet, get_num_actions
from torch import nn as nn
from transformers import ViTFeatureExtractor, ViTMAEForPreTraining

if TYPE_CHECKING:
    from omegaconf import DictConfig

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric_temporal.nn.recurrent import GConvGRU
from transformers import ViTConfig, ViTFeatureExtractor, ViTMAEForPreTraining


class vitmae:
    def __init__(self):
        model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "facebook/vit-mae-base"
        )

        self.encoder = model.vit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(self.device)
        # self.encoder.to(self.device)
        self.encoder.eval()

        print("vitmae initialized TEST ")

    def forward(self, observation):
        x = observation["rgb"]
        x = self.feature_extractor(images=x, return_tensors="pt").pixel_values
        # x = x.to(self.device)
        embed = self.encoder(x).last_hidden_state[:, 0]
        embed = embed.to(self.device)
        return embed

    @property
    def is_blind(self):
        return False


class Policy(abc.ABC):
    action_distribution: nn.Module

    def __init__(self):
        pass

    @property
    def should_load_agent_state(self):
        return True

    @property
    def num_recurrent_layers(self) -> int:
        return 0

    def forward(self, *x):
        raise NotImplementedError

    def get_policy_info(self, infos, dones) -> List[Dict[str, float]]:
        """
        Gets the log information from the policy at the current time step.
        Currently only called during evaluation. The return list should be
        empty for no logging or a list of size equal to the number of
        environments.
        """

        return []

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        pass


class NetPolicy(nn.Module, Policy):
    aux_loss_modules: nn.ModuleDict

    def __init__(self, net, action_space, policy_config=None, aux_loss_config=None):
        super().__init__()
        self.net = net
        self.dim_actions = get_num_actions(action_space)
        self.action_distribution: Union[CategoricalNet, GaussianNet]

        if policy_config is None:
            self.action_distribution_type = "categorical"
        else:
            self.action_distribution_type = policy_config.action_distribution_type

        if self.action_distribution_type == "categorical":
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )
        elif self.action_distribution_type == "gaussian":
            self.action_distribution = GaussianNet(
                self.net.output_size,
                self.dim_actions,
                policy_config.action_dist,
            )
        else:
            raise ValueError(
                f"Action distribution {self.action_distribution_type}" "not supported."
            )

        self.critic = CriticHead(self.net.output_size)

        self.aux_loss_modules = nn.ModuleDict()
        if aux_loss_config is None:
            return
        for aux_loss_name, cfg in aux_loss_config.items():
            aux_loss = baseline_registry.get_auxiliary_loss(aux_loss_name)

            self.aux_loss_modules[aux_loss_name] = aux_loss(
                action_space,
                self.net,
                **cfg,
            )

    @property
    def should_load_agent_state(self):
        return True

    @property
    def num_recurrent_layers(self) -> int:
        return self.net.num_recurrent_layers

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ):
        features, rnn_hidden_states, aux_loss_state = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info,
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        batch = dict(
            observations=observations,
            rnn_hidden_states=rnn_hidden_states,
            prev_actions=prev_actions,
            masks=masks,
            action=action,
            rnn_build_seq_info=rnn_build_seq_info,
        )
        aux_loss_res = {
            k: v(aux_loss_state, batch) for k, v in self.aux_loss_modules.items()
        }

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_loss_res,
        )

    @property
    def policy_components(self):
        return (self.net, self.critic, self.action_distribution)

    def policy_parameters(self) -> Iterable[torch.Tensor]:
        for c in self.policy_components:
            yield from c.parameters()

    def all_policy_tensors(self) -> Iterable[torch.Tensor]:
        yield from self.policy_parameters()
        for c in self.policy_components:
            yield from c.buffers()

    def aux_loss_parameters(self) -> Dict[str, Iterable[torch.Tensor]]:
        return {k: v.parameters() for k, v in self.aux_loss_modules.items()}

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


@baseline_registry.register_policy
class PointNavBaselinePolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        aux_loss_config=None,
        **kwargs,
    ):
        super().__init__(
            PointNavBaselineNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                **kwargs,
            ),
            action_space=action_space,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass

    @property
    @abc.abstractmethod
    def perception_embedding_size(self) -> int:
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
    ):
        super().__init__()

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces:
            self._n_input_goal = observation_space.spaces[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ].shape[0]
        elif PointGoalSensor.cls_uuid in observation_space.spaces:
            self._n_input_goal = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
        elif ImageGoalSensor.cls_uuid in observation_space.spaces:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = SimpleCNN(goal_observation_space, hidden_size)
            # self.goal_visual_encoder = vitmae()
            self._n_input_goal = hidden_size

        self._hidden_size = hidden_size

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.graph_encoder = GCN(
            517,
            512,
            hidden_size,
        )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size)
            + self._n_input_goal
            + hidden_size,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def forward(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ):
        aux_loss_state = {}
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            target_encoding = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
        elif PointGoalSensor.cls_uuid in observations:
            target_encoding = observations[PointGoalSensor.cls_uuid]
        elif ImageGoalSensor.cls_uuid in observations:
            image_goal = observations[ImageGoalSensor.cls_uuid]
            target_encoding = self.goal_visual_encoder({"rgb": image_goal})

        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x
            aux_loss_state["perception_embed"] = perception_embed

        x_out = torch.cat(x, dim=1)

        # Add the graph network
        batch_size = x_out.shape[0]
        train_dataset = []

        for i in range(batch_size):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ring_network = RingAttractorNetworkGraph(512)  # (self.nb_of_nodes)

            # get image encoding for the current image
            node_feat = ring_network.get_node_features(x_out[i], 2, odometry=None)

            # Set the node and edge features in the graph object
            for i, feat in enumerate(node_feat):
                ring_network.ran_graph.nodes[i]["feature"] = feat

            # Generate the input features and edge index tensor for the model
            x = torch.tensor(node_feat, dtype=torch.float)

            edge_index = ring_network.get_edge_index()
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, edge_attr=ring_network.edge_feat)

            train_dataset.append(data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Compute the graph embedding using the GCN model
        # We will only have one iteration since the dataset is always of shape
        # batch size
        embedding = torch.zeros((batch_size, self._hidden_size))
        for data in train_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.batch = data.batch.to(device)
            embedding = self.state_encoder(data.x, data.edge_index, data.batch)

        # embedding is of shape (batch_size, hidden_size of CNN)
        print("embedding shape: ", embedding.shape)
        print("x_out shape: ", x_out.shape)

        x_out = torch.cat((x_out, embedding), dim=1)

        # Add the graph embedding to the input
        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = x_out

        return x_out, rnn_hidden_states, aux_loss_state


#################### Added for the graph network ####################
# Our own defined network
class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GConvGRU(num_features, hidden_channels, K=2)
        self.conv2 = GConvGRU(hidden_channels, out_channels, K=2)

    def forward(self, x, edge_index, batch):
        # Apply the first GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Apply the second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

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
        """_summary_

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
        """_summary_

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
