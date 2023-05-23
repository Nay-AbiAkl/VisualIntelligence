#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

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

if TYPE_CHECKING:
    from omegaconf import DictConfig

import time
from typing import Optional

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
from transformers import ViTConfig, ViTFeatureExtractor, ViTMAEForPreTraining


class vitmae:
    def __init__(self):
        model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "facebook/vit-mae-base"
        )
        self.encoder = model.vit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()
        print("vitmae initialized TEST ")

    def forward(self, observation):
        observation.to(self.device)
        x = observation["rgb"]
        x = self.feature_extractor(images=x, return_tensors="pt").pixel_values
        embed = self.encoder(x).last_hidden_state[:, 0]
        return embed

    @property
    def is_blind(self):
        return False


@dataclass
class PolicyActionData:
    """
    Information returned from the `Policy.act` method representing the
    information from an agent's action.

    :property should_inserts: Of shape [# envs, 1]. If False at environment
        index `i`, then don't write this transition to the rollout buffer. If
        `None`, then write all data.
    :property policy_info`: Optional logging information about the policy per
        environment. For example, you could log the policy entropy.
    :property take_actions`: If specified, these actions will be executed in
        the environment, but not stored in the storage buffer. This allows
        exectuing and learning from different actions. If not specified, the
        agent will execute `self.actions`.
    :property values: The actor value predictions. None if the actor does not predict value.
    :property actions: The actions to store in the storage buffer. if
        `take_actions` is None, then this is also the action executed in the
        environment.
    :property rnn_hidden_states: Actor hidden states.
    :property action_log_probs: The log probabilities of the actions under the
        current policy.
    """

    actions: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    action_log_probs: Optional[torch.Tensor] = None
    take_actions: Optional[torch.Tensor] = None
    policy_info: Optional[List[Dict[str, Any]]] = None
    should_inserts: Optional[torch.BoolTensor] = None

    def write_action(self, write_idx: int, write_action: torch.Tensor) -> None:
        """
        Used to override an action across all environments.
        :param write_idx: The index in the action dimension to write the new action.
        :param write_action: The action to write at `write_idx`.
        """
        self.actions[:, write_idx] = write_action

    @property
    def env_actions(self) -> torch.Tensor:
        """
        The actions to execute in the environment.
        """

        if self.take_actions is None:
            return self.actions
        else:
            return self.take_actions


class Policy(abc.ABC):
    action_distribution: nn.Module

    def __init__(self):
        pass

    @property
    def should_load_agent_state(self):
        return True

    @property
    def num_hidden_channels(self) -> int:
        return 0

    def forward(self, *x):
        raise NotImplementedError

    def get_policy_action_space(self, env_action_space: spaces.Space) -> spaces.Space:
        return env_action_space

    def _get_policy_components(self) -> List[nn.Module]:
        return []

    def aux_loss_parameters(self) -> Dict[str, Iterable[torch.Tensor]]:
        return {}

    def policy_parameters(self) -> Iterable[torch.Tensor]:
        for c in self._get_policy_components():
            yield from c.parameters()

    def all_policy_tensors(self) -> Iterable[torch.Tensor]:
        yield from self.policy_parameters()
        for c in self._get_policy_components():
            yield from c.buffers()

    def get_value(self, observations) -> torch.Tensor:
        raise NotImplementedError(
            "Get value is supported in non-neural network policies."
        )

    def get_extra(
        self, action_data: PolicyActionData, infos, dones
    ) -> List[Dict[str, float]]:
        """
        Gets the log information from the policy at the current time step.
        Currently only called during evaluation. The return list should be
        empty for no logging or a list of size equal to the number of
        environments.
        """
        if action_data.policy_info is None:
            return []
        else:
            return action_data.policy_info

    def act(
        self,
        observations,
        deterministic=False,
    ) -> PolicyActionData:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        # state_encoder_input_channels,
        # state_encoder_hidden_channels,
        # state_encoder_out_channels,
        # nb_of_nodes,
        **kwargs,
    ):
        pass


# This class we will need
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
    def num_hidden_channels(self) -> int:
        return self.net.num_hidden_channels

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        deterministic=False,
    ):
        features, _ = self.net(observations)

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
        return PolicyActionData(
            values=value,
            actions=action,
            action_log_probs=action_log_probs,
        )

    def get_value(self, observations):
        features, _ = self.net(observations)
        return self.critic(features)

    def evaluate_actions(
        self,
        observations,
        recurrent_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ):
        features, aux_loss_state = self.net(
            observations,
            # rnn_build_seq_info,
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        batch = dict(
            observations=observations,
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
            0,
            aux_loss_res,
        )

    def _get_policy_components(self) -> List[nn.Module]:
        return [self.net, self.critic, self.action_distribution]

    def aux_loss_parameters(self) -> Dict[str, Iterable[torch.Tensor]]:
        return {k: v.parameters() for k, v in self.aux_loss_modules.items()}

    @classmethod
    @abc.abstractmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        # state_encoder_input_channels,
        # state_encoder_hidden_channels,
        # state_encoder_out_channels,
        # nb_of_nodes,
        **kwargs,
    ):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


# @baseline_registry.register_policy
# class PointNavBaselinePolicy(NetPolicy):
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         action_space,
#         hidden_size: int = 512,
#         aux_loss_config=None,
#         **kwargs,
#     ):
#         super().__init__(
#             PointNavBaselineNet(  # type: ignore
#                 observation_space=observation_space,
#                 hidden_size=hidden_size,
#                 **kwargs,
#             ),
#             action_space=action_space,
#             aux_loss_config=aux_loss_config,
#         )

#     @classmethod
#     def from_config(
#         cls,
#         config: "DictConfig",
#         observation_space: spaces.Dict,
#         action_space,
#         **kwargs,
#     ):
#         return cls(
#             observation_space=observation_space,
#             action_space=action_space,
#             hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
#             aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
#         )


@baseline_registry.register_ran_policy
class GCNPointNavBaselinePolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        state_encoder_input_channels=771,
        state_encoder_hidden_channels=512,
        state_encoder_out_channels=512,
        nb_of_nodes=100,
        hidden_size: int = 512,
        aux_loss_config=None,
        **kwargs,
    ):
        super().__init__(
            GCNPointNavBaselineNet(  # type: ignore
                observation_space=observation_space,
                state_encoder_input_channels=state_encoder_input_channels,
                state_encoder_hidden_channels=state_encoder_hidden_channels,
                state_encoder_out_channels=state_encoder_out_channels,
                nb_of_nodes=nb_of_nodes,
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
            state_encoder_input_channels=config.habitat_baselines.rl.policy.state_encoder_input_channels,
            state_encoder_hidden_channels=config.habitat_baselines.rl.policy.state_encoder_hidden_channels,
            state_encoder_out_channels=config.habitat_baselines.rl.policy.state_encoder_out_channels,
            nb_of_nodes=config.habitat_baselines.rl.policy.nb_of_nodes,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
        )


# we will not use this base class for the network, replace it with the GCN network
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
    def num_hidden_channels(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
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
            self._n_input_goal = hidden_size

        self._hidden_size = hidden_size

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
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
            target_encoding = self.goal_visual_encoder.forward({"rgb": image_goal})

        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder.forward(observations)
            x = [perception_embed] + x
            aux_loss_state["perception_embed"] = perception_embed

        x_out = torch.cat(x, dim=1)
        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = x_out

        return x_out, rnn_hidden_states, aux_loss_state


# Our own defined network
class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # add 5 fully connected layers to gradually reduce the dimension
        # self.fc1 = torch.nn.Linear(out_channels, 64)
        # self.fc2 = torch.nn.Linear(64, 32)
        # self.fc3 = torch.nn.Linear(32, 16)
        # self.fc4 = torch.nn.Linear(16, 8)
        # self.fc5 = torch.nn.Linear(8, 4)

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

        # add fully connected layers
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = self.fc5(x)

        return x


class RingAttractorNetworkGraph:
    """Graph representing the ring attractor network"""

    def __init__(self, nb_of_nodes: int) -> None:
        # Note that final center node is of idx nb_of_nodes-1
        print("RAN GRAPH INIT")
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


class GCNPointNavBaselineNet(Net):
    r"""Graph Convolutional Network for PointNav baseline"""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_encoder_input_channels: int,
        state_encoder_hidden_channels: int,
        state_encoder_out_channels: int,
        hidden_size: int,
        nb_of_nodes: int,
        simple_cnn: bool = False,
    ):
        super().__init__()
        print("GCN INIT")

        self.state_encoder_input_channels = state_encoder_input_channels
        self.state_encoder_hidden_channels = state_encoder_hidden_channels
        self.state_encoder_out_channels = state_encoder_out_channels
        self.nb_of_nodes = nb_of_nodes

        # self.ring_network = RingAttractorNetworkGraph(self.nb_of_nodes)

        self.state_encoder = GCN(
            state_encoder_input_channels,
            state_encoder_hidden_channels,
            state_encoder_out_channels,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_encoder.to(device=device)

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
            # self.goal_visual_encoder = SimpleCNN(goal_observation_space, hidden_size)
            if simple_cnn:
                self.goal_visual_encoder = SimpleCNN(
                    goal_observation_space, hidden_size
                )
            else:
                self.goal_visual_encoder = vitmae()

            self._n_input_goal = hidden_size

        self._hidden_size = hidden_size
        if simple_cnn:
            self.visual_encoder = SimpleCNN(observation_space, hidden_size)
        else:
            self.visual_encoder = vitmae()

        self.train()

    @property
    def output_size(self):
        return (
            self.state_encoder_hidden_channels
            # 4
            # state encoder hidden channels is another option
        )

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_hidden_channels(self):
        return self.state_encoder_hidden_channels

    @property
    def num_recurrent_layers(self):
        return 0

    def forward(
        self,
        observations,
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
            target_encoding = self.goal_visual_encoder.foward({"rgb": image_goal})

        x = [target_encoding]
        # print("obsevrations", observations["rgb"].shape)
        if not self.is_blind:
            perception_embed = self.visual_encoder.forward(observations)
            x = [perception_embed] + x
            aux_loss_state["perception_embed"] = perception_embed

        # x_out = torch.cat(x, dim=1)

        image_encoding = perception_embed

        # batch size is the first dimension in the image encoding
        batch_size = image_encoding.shape[0]
        train_dataset = []

        for i in range(batch_size):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ring_network = RingAttractorNetworkGraph(self.nb_of_nodes)

            # get image encoding for the current image
            node_feat = ring_network.get_node_features(
                image_encoding[i], 2, odometry=None
            )

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
        # We will only have one iteration since the dataset is always of shape batch size
        for data in train_loader:
            embedding = self.state_encoder(data.x, data.edge_index, data.batch)

        return embedding, aux_loss_state


#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# import abc
# from dataclasses import dataclass
# from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

# import torch
# from gym import spaces
# from habitat.tasks.nav.nav import (
#     ImageGoalSensor,
#     IntegratedPointGoalGPSAndCompassSensor,
#     PointGoalSensor,
# )
# from habitat_baselines.common.baseline_registry import baseline_registry
# from habitat_baselines.rl.models.rnn_state_encoder import (
#     build_rnn_state_encoder,
# )
# from habitat_baselines.rl.models.simple_cnn import SimpleCNN
# from habitat_baselines.utils.common import (
#     CategoricalNet,
#     GaussianNet,
#     get_num_actions,
# )
# from torch import nn as nn

# if TYPE_CHECKING:
#     from omegaconf import DictConfig

# import time
# from typing import Optional

# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data as utils
# import torch_geometric
# from torch import nn, optim
# from torch.utils.data.sampler import SubsetRandomSampler
# from torch_geometric.nn import GCNConv, global_mean_pool
# from transformers import ViTConfig, ViTFeatureExtractor, ViTMAEForPreTraining


# class vitmae:
#     def __init__(self):
#         model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
#         self.feature_extractor = ViTFeatureExtractor.from_pretrained(
#             "facebook/vit-mae-base"
#         )
#         self.encoder = model.vit
#         self.encoder.eval()
#         print("vitmae initialized")

#     def forward(self, observation):
#         x = observation["rgb"]
#         x = self.feature_extractor(images=x, return_tensors="pt").pixel_values
#         embed = self.encoder(x).last_hidden_state[:, 0]
#         return embed

#     @property
#     def is_blind(self):
#         return False


# @dataclass
# class PolicyActionData:
#     """
#     Information returned from the `Policy.act` method representing the
#     information from an agent's action.

#     :property should_inserts: Of shape [# envs, 1]. If False at environment
#         index `i`, then don't write this transition to the rollout buffer. If
#         `None`, then write all data.
#     :property policy_info`: Optional logging information about the policy per
#         environment. For example, you could log the policy entropy.
#     :property take_actions`: If specified, these actions will be executed in
#         the environment, but not stored in the storage buffer. This allows
#         exectuing and learning from different actions. If not specified, the
#         agent will execute `self.actions`.
#     :property values: The actor value predictions. None if the actor does not predict value.
#     :property actions: The actions to store in the storage buffer. if
#         `take_actions` is None, then this is also the action executed in the
#         environment.
#     :property rnn_hidden_states: Actor hidden states.
#     :property action_log_probs: The log probabilities of the actions under the
#         current policy.
#     """

#     actions: Optional[torch.Tensor] = None
#     values: Optional[torch.Tensor] = None
#     action_log_probs: Optional[torch.Tensor] = None
#     take_actions: Optional[torch.Tensor] = None
#     policy_info: Optional[List[Dict[str, Any]]] = None
#     should_inserts: Optional[torch.BoolTensor] = None

#     def write_action(self, write_idx: int, write_action: torch.Tensor) -> None:
#         """
#         Used to override an action across all environments.
#         :param write_idx: The index in the action dimension to write the new action.
#         :param write_action: The action to write at `write_idx`.
#         """
#         self.actions[:, write_idx] = write_action

#     @property
#     def env_actions(self) -> torch.Tensor:
#         """
#         The actions to execute in the environment.
#         """

#         if self.take_actions is None:
#             return self.actions
#         else:
#             return self.take_actions


# class Policy(abc.ABC):
#     action_distribution: nn.Module

#     def __init__(self):
#         pass

#     @property
#     def should_load_agent_state(self):
#         return True

#     @property
#     def num_hidden_channels(self) -> int:
#         return 0

#     def forward(self, *x):
#         raise NotImplementedError

#     def get_policy_action_space(
#         self, env_action_space: spaces.Space
#     ) -> spaces.Space:
#         return env_action_space

#     def _get_policy_components(self) -> List[nn.Module]:
#         return []

#     def aux_loss_parameters(self) -> Dict[str, Iterable[torch.Tensor]]:
#         return {}

#     def policy_parameters(self) -> Iterable[torch.Tensor]:
#         for c in self._get_policy_components():
#             yield from c.parameters()

#     def all_policy_tensors(self) -> Iterable[torch.Tensor]:
#         yield from self.policy_parameters()
#         for c in self._get_policy_components():
#             yield from c.buffers()

#     def get_value(self, observations) -> torch.Tensor:
#         raise NotImplementedError(
#             "Get value is supported in non-neural network policies."
#         )

#     def get_extra(
#         self, action_data: PolicyActionData, infos, dones
#     ) -> List[Dict[str, float]]:
#         """
#         Gets the log information from the policy at the current time step.
#         Currently only called during evaluation. The return list should be
#         empty for no logging or a list of size equal to the number of
#         environments.
#         """
#         if action_data.policy_info is None:
#             return []
#         else:
#             return action_data.policy_info

#     def act(
#         self,
#         observations,
#         deterministic=False,
#     ) -> PolicyActionData:
#         raise NotImplementedError

#     @classmethod
#     @abc.abstractmethod
#     def from_config(
#         cls,
#         config,
#         observation_space,
#         action_space,
#         # state_encoder_input_channels,
#         # state_encoder_hidden_channels,
#         # state_encoder_out_channels,
#         # nb_of_nodes,
#         **kwargs,
#     ):
#         pass


# # This class we will need
# class NetPolicy(nn.Module, Policy):
#     aux_loss_modules: nn.ModuleDict

#     def __init__(
#         self, net, action_space, policy_config=None, aux_loss_config=None
#     ):
#         super().__init__()
#         self.net = net
#         self.dim_actions = get_num_actions(action_space)
#         self.action_distribution: Union[CategoricalNet, GaussianNet]

#         if policy_config is None:
#             self.action_distribution_type = "categorical"
#         else:
#             self.action_distribution_type = (
#                 policy_config.action_distribution_type
#             )

#         if self.action_distribution_type == "categorical":
#             self.action_distribution = CategoricalNet(
#                 self.net.output_size, self.dim_actions
#             )
#         elif self.action_distribution_type == "gaussian":
#             self.action_distribution = GaussianNet(
#                 self.net.output_size,
#                 self.dim_actions,
#                 policy_config.action_dist,
#             )
#         else:
#             raise ValueError(
#                 f"Action distribution {self.action_distribution_type}"
#                 "not supported."
#             )

#         self.critic = CriticHead(self.net.output_size)

#         self.aux_loss_modules = nn.ModuleDict()
#         if aux_loss_config is None:
#             return
#         for aux_loss_name, cfg in aux_loss_config.items():
#             aux_loss = baseline_registry.get_auxiliary_loss(aux_loss_name)

#             self.aux_loss_modules[aux_loss_name] = aux_loss(
#                 action_space,
#                 self.net,
#                 **cfg,
#             )

#     @property
#     def should_load_agent_state(self):
#         return True

#     @property
#     def num_hidden_channels(self) -> int:
#         return self.net.num_hidden_channels

#     def forward(self, *x):
#         raise NotImplementedError

#     def act(
#         self,
#         observations,
#         deterministic=False,
#     ):
#         features, _ = self.net(observations)

#         distribution = self.action_distribution(features)
#         value = self.critic(features)

#         if deterministic:
#             if self.action_distribution_type == "categorical":
#                 action = distribution.mode()
#             elif self.action_distribution_type == "gaussian":
#                 action = distribution.mean
#         else:
#             action = distribution.sample()

#         action_log_probs = distribution.log_probs(action)
#         return PolicyActionData(
#             values=value,
#             actions=action,
#             action_log_probs=action_log_probs,
#         )

#     def get_value(self, observations):
#         features, _ = self.net(observations)
#         return self.critic(features)

#     def evaluate_actions(
#         self,
#         observations,
#         recurrent_hidden_states,
#         prev_actions,
#         masks,
#         action,
#         rnn_build_seq_info: Dict[str, torch.Tensor],
#     ):
#         features, aux_loss_state = self.net(
#             observations,
#             # rnn_build_seq_info,
#         )
#         distribution = self.action_distribution(features)
#         value = self.critic(features)

#         action_log_probs = distribution.log_probs(action)
#         distribution_entropy = distribution.entropy()

#         batch = dict(
#             observations=observations,
#             action=action,
#             rnn_build_seq_info=rnn_build_seq_info,
#         )
#         aux_loss_res = {
#             k: v(aux_loss_state, batch)
#             for k, v in self.aux_loss_modules.items()
#         }

#         return (
#             value,
#             action_log_probs,
#             distribution_entropy,
#             0,
#             aux_loss_res,
#         )

#     def _get_policy_components(self) -> List[nn.Module]:
#         return [self.net, self.critic, self.action_distribution]

#     def aux_loss_parameters(self) -> Dict[str, Iterable[torch.Tensor]]:
#         return {k: v.parameters() for k, v in self.aux_loss_modules.items()}

#     @classmethod
#     @abc.abstractmethod
#     def from_config(
#         cls,
#         config,
#         observation_space,
#         action_space,
#         # state_encoder_input_channels,
#         # state_encoder_hidden_channels,
#         # state_encoder_out_channels,
#         # nb_of_nodes,
#         **kwargs,
#     ):
#         pass


# class CriticHead(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#         self.fc = nn.Linear(input_size, 1)
#         nn.init.orthogonal_(self.fc.weight)
#         nn.init.constant_(self.fc.bias, 0)

#     def forward(self, x):
#         return self.fc(x)


# # @baseline_registry.register_policy
# # class PointNavBaselinePolicy(NetPolicy):
# #     def __init__(
# #         self,
# #         observation_space: spaces.Dict,
# #         action_space,
# #         hidden_size: int = 512,
# #         aux_loss_config=None,
# #         **kwargs,
# #     ):
# #         super().__init__(
# #             PointNavBaselineNet(  # type: ignore
# #                 observation_space=observation_space,
# #                 hidden_size=hidden_size,
# #                 **kwargs,
# #             ),
# #             action_space=action_space,
# #             aux_loss_config=aux_loss_config,
# #         )

# #     @classmethod
# #     def from_config(
# #         cls,
# #         config: "DictConfig",
# #         observation_space: spaces.Dict,
# #         action_space,
# #         **kwargs,
# #     ):
# #         return cls(
# #             observation_space=observation_space,
# #             action_space=action_space,
# #             hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
# #             aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
# #         )


# @baseline_registry.register_ran_policy
# class GCNPointNavBaselinePolicy(NetPolicy):
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         action_space,
#         state_encoder_input_channels=771,
#         state_encoder_hidden_channels=512,
#         state_encoder_out_channels=512,
#         nb_of_nodes=100,
#         hidden_size: int = 512,
#         aux_loss_config=None,
#         **kwargs,
#     ):
#         super().__init__(
#             GCNPointNavBaselineNet(  # type: ignore
#                 observation_space=observation_space,
#                 state_encoder_input_channels=state_encoder_input_channels,
#                 state_encoder_hidden_channels=state_encoder_hidden_channels,
#                 state_encoder_out_channels=state_encoder_out_channels,
#                 nb_of_nodes=nb_of_nodes,
#                 hidden_size=hidden_size,
#                 **kwargs,
#             ),
#             action_space=action_space,
#             aux_loss_config=aux_loss_config,
#         )

#     @classmethod
#     def from_config(
#         cls,
#         config: "DictConfig",
#         observation_space: spaces.Dict,
#         action_space,
#         **kwargs,
#     ):
#         return cls(
#             observation_space=observation_space,
#             action_space=action_space,
#             state_encoder_input_channels=config.habitat_baselines.rl.policy.state_encoder_input_channels,
#             state_encoder_hidden_channels=config.habitat_baselines.rl.policy.state_encoder_hidden_channels,
#             state_encoder_out_channels=config.habitat_baselines.rl.policy.state_encoder_out_channels,
#             nb_of_nodes=config.habitat_baselines.rl.policy.nb_of_nodes,
#             hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
#             aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
#         )


# # we will not use this base class for the network, replace it with the GCN network
# class Net(nn.Module, metaclass=abc.ABCMeta):
#     @abc.abstractmethod
#     def forward(self, observations, rnn_hidden_states, prev_actions, masks):
#         pass

#     @property
#     @abc.abstractmethod
#     def output_size(self):
#         pass

#     @property
#     @abc.abstractmethod
#     def num_hidden_channels(self):
#         pass

#     @property
#     @abc.abstractmethod
#     def is_blind(self):
#         pass


# class PointNavBaselineNet(Net):
#     r"""Network which passes the input image through CNN and concatenates
#     goal vector with CNN's output and passes that through RNN.
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         hidden_size: int,
#     ):
#         super().__init__()

#         if (
#             IntegratedPointGoalGPSAndCompassSensor.cls_uuid
#             in observation_space.spaces
#         ):
#             self._n_input_goal = observation_space.spaces[
#                 IntegratedPointGoalGPSAndCompassSensor.cls_uuid
#             ].shape[0]
#         elif PointGoalSensor.cls_uuid in observation_space.spaces:
#             self._n_input_goal = observation_space.spaces[
#                 PointGoalSensor.cls_uuid
#             ].shape[0]
#         elif ImageGoalSensor.cls_uuid in observation_space.spaces:
#             goal_observation_space = spaces.Dict(
#                 {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
#             )
#             self.goal_visual_encoder = SimpleCNN(
#                 goal_observation_space, hidden_size
#             )
#             self._n_input_goal = hidden_size

#         self._hidden_size = hidden_size

#         self.visual_encoder = SimpleCNN(observation_space, hidden_size)

#         self.state_encoder = build_rnn_state_encoder(
#             (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
#             self._hidden_size,
#         )

#         self.train()

#     @property
#     def output_size(self):
#         return self._hidden_size

#     @property
#     def is_blind(self):
#         return self.visual_encoder.is_blind

#     @property
#     def num_recurrent_layers(self):
#         return self.state_encoder.num_recurrent_layers

#     @property
#     def perception_embedding_size(self):
#         return self._hidden_size

#     def forward(
#         self,
#         observations,
#         rnn_hidden_states,
#         prev_actions,
#         masks,
#         rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
#     ):
#         aux_loss_state = {}
#         if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
#             target_encoding = observations[
#                 IntegratedPointGoalGPSAndCompassSensor.cls_uuid
#             ]
#         elif PointGoalSensor.cls_uuid in observations:
#             target_encoding = observations[PointGoalSensor.cls_uuid]
#         elif ImageGoalSensor.cls_uuid in observations:
#             image_goal = observations[ImageGoalSensor.cls_uuid]
#             target_encoding = self.goal_visual_encoder.forward(
#                 {"rgb": image_goal}
#             )

#         x = [target_encoding]

#         if not self.is_blind:
#             perception_embed = self.visual_encoder.forward(observations)
#             x = [perception_embed] + x
#             aux_loss_state["perception_embed"] = perception_embed

#         x_out = torch.cat(x, dim=1)
#         x_out, rnn_hidden_states = self.state_encoder(
#             x_out, rnn_hidden_states, masks, rnn_build_seq_info
#         )
#         aux_loss_state["rnn_output"] = x_out

#         return x_out, rnn_hidden_states, aux_loss_state


# # Our own defined network
# class GCN(nn.Module):
#     def __init__(self, num_features, hidden_channels, out_channels):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#         # add 5 fully connected layers to gradually reduce the dimension
#         self.fc1 = torch.nn.Linear(out_channels, 64)
#         self.fc2 = torch.nn.Linear(64, 32)
#         self.fc3 = torch.nn.Linear(32, 16)
#         self.fc4 = torch.nn.Linear(16, 8)
#         self.fc5 = torch.nn.Linear(8, 4)

#     def forward(self, x, edge_index):
#         # Apply the first GCN layer
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)

#         # Apply the second GCN layer
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)

#         # Global pooling to obtain the graph embedding
#         x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))

#         # add fully connected layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc5(x)

#         return x


# class RingAttractorNetworkGraph:
#     """Graph representing the ring attractor network"""

#     def __init__(self, nb_of_nodes: int) -> None:
#         # Note that final center node is of idx nb_of_nodes-1
#         self.nb_of_nodes = nb_of_nodes
#         self.ran_graph = nx.MultiDiGraph()

#         # Add nb_of_nodes nodes
#         nodes = range(0, self.nb_of_nodes - 1)
#         self.ran_graph.add_nodes_from(nodes)

#         # Add bidirectional connections for the first nb_of_nodes-1 nodes
#         for i in range(0, self.nb_of_nodes - 2):
#             self.ran_graph.add_edge(i, i + 1)
#             self.ran_graph.add_edge(i + 1, i)

#         # Add connections between idx nb_of_nodes-2 and 0

#         self.ran_graph.add_edge(self.nb_of_nodes - 2, 0)
#         self.ran_graph.add_edge(0, self.nb_of_nodes - 2)

#         # Add unidirectional connections for the last node
#         for i in range(0, self.nb_of_nodes - 1):
#             self.ran_graph.add_edge(self.nb_of_nodes - 1, i)

#         # add edge features
#         edge_feat = self.get_edge_features()
#         for idx, (_, _, e) in enumerate(self.ran_graph.edges(data=True)):
#             e["feature"] = edge_feat[idx]

#     # Function to prepare the node features
#     def get_node_features(
#         self,
#         image_encoding: torch.Tensor,
#         nb_connections: int = 2,  # TODO: CHANGE TO OPTIONAL
#         odometry: Optional[torch.Tensor] = None,
#     ):
#         """_summary_

#         Args:
#             nb_nodes (int): total nb of nodes (including the center node), odd number
#             image_encoding (torch.Tensor): embeddings of the image
#             node_pos_encoding (_type_): circular positional encoding
#             nb_connections (Optional[int], optional): nb of connections of each node.
#             Defaults to 2.
#             odometry (Optional[torch.Tensor], optional): odom data Defaults to None.
#         """
#         positions = self.distribute_nodes_on_circle(radius=1)

#         nb_connections_tensor = torch.tensor(nb_connections).unsqueeze(0)
#         nb_connections_tensor = nb_connections_tensor.repeat(
#             self.nb_of_nodes - 1, 1
#         )
#         nb_connections_tensor = torch.cat(
#             (
#                 nb_connections_tensor,
#                 torch.tensor([self.nb_of_nodes - 1]).unsqueeze(0),
#             ),
#             0,
#         )
#         # print("before repeat", image_encoding.shape)
#         image_encoding = image_encoding.repeat(self.nb_of_nodes, 1)

#         feature_vector = torch.concat(
#             (image_encoding, nb_connections_tensor, positions), 1
#         )
#         # print("before repeat", image_encoding.shape)
#         if odometry is not None:
#             odometry = odometry.repeat(self.nb_of_nodes, 1)
#             feature_vector = torch.concat((feature_vector, odometry), 1)

#         return feature_vector

#     # Function to prepare the edge features
#     def get_edge_features(self):
#         """_summary_

#         Args:
#             nb_nodes (int): total nb of nodes (including the center node), odd number
#         """
#         # Define the edge features tensor with shape (2 * nb_nodes + nb_nodes, 1)
#         # first 2 * nb_nodes are for the connections between the nodes, should be 1
#         # last nb_nodes are for the connections between the new node and the existing
#         # nodes, should be -1
#         edge_features = torch.ones(
#             (2 * (self.nb_of_nodes - 1) + (self.nb_of_nodes - 1), 1)
#         )
#         edge_features[2 * (self.nb_of_nodes - 1) :, :] = -1

#         return edge_features

#     def get_edge_index(self):
#         # create edge index from
#         adj = nx.to_scipy_sparse_array(self.ran_graph).tocoo()
#         row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
#         col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
#         edge_index = torch.stack([row, col], dim=0)

#         return edge_index

#     def distribute_nodes_on_circle(self, radius=1):
#         angles = np.linspace(
#             0, 2 * np.pi, self.nb_of_nodes - 1, endpoint=False
#         )  # Divide the circle into equal angles
#         x = radius * np.cos(
#             angles
#         )  # Calculate x-coordinates using cosine function
#         y = radius * np.sin(
#             angles
#         )  # Calculate y-coordinates using sine function
#         positions = np.column_stack(
#             (x, y)
#         )  # Stack x and y coordinates as column vectors

#         positions = np.vstack((positions, np.array([0, 0])))
#         return torch.tensor(positions)


# class GCNPointNavBaselineNet(Net):
#     r"""Graph Convolutional Network for PointNav baseline"""

#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         state_encoder_input_channels: int,
#         state_encoder_hidden_channels: int,
#         state_encoder_out_channels: int,
#         hidden_size: int,
#         nb_of_nodes: int,
#         simple_cnn: bool = False,
#     ):
#         super().__init__()

#         self.state_encoder_input_channels = state_encoder_input_channels
#         self.state_encoder_hidden_channels = state_encoder_hidden_channels
#         self.state_encoder_out_channels = state_encoder_out_channels
#         self.nb_of_nodes = nb_of_nodes

#         self.ring_network = RingAttractorNetworkGraph(self.nb_of_nodes)

#         self.state_encoder = GCN(
#             state_encoder_input_channels,
#             state_encoder_hidden_channels,
#             state_encoder_out_channels,
#         )

#         if (
#             IntegratedPointGoalGPSAndCompassSensor.cls_uuid
#             in observation_space.spaces
#         ):
#             self._n_input_goal = observation_space.spaces[
#                 IntegratedPointGoalGPSAndCompassSensor.cls_uuid
#             ].shape[0]
#         elif PointGoalSensor.cls_uuid in observation_space.spaces:
#             self._n_input_goal = observation_space.spaces[
#                 PointGoalSensor.cls_uuid
#             ].shape[0]
#         elif ImageGoalSensor.cls_uuid in observation_space.spaces:
#             goal_observation_space = spaces.Dict(
#                 {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
#             )
#             # self.goal_visual_encoder = SimpleCNN(goal_observation_space, hidden_size)
#             if simple_cnn:
#                 self.goal_visual_encoder = SimpleCNN(
#                     goal_observation_space, hidden_size
#                 )
#             else:
#                 self.goal_visual_encoder = vitmae()

#             self._n_input_goal = hidden_size

#         self._hidden_size = hidden_size
#         if simple_cnn:
#             self.visual_encoder = SimpleCNN(observation_space, hidden_size)
#         else:
#             self.visual_encoder = vitmae()

#         self.train()

#     @property
#     def output_size(self):
#         return (
#             4
#             # self.state_encoder_out_channels
#             # state encoder hidden channels is another option
#         )

#     @property
#     def is_blind(self):
#         return self.visual_encoder.is_blind

#     @property
#     def num_hidden_channels(self):
#         return self.state_encoder_hidden_channels

#     @property
#     def num_recurrent_layers(self):
#         return 0

#     def forward(
#         self,
#         observations,
#     ):
#         aux_loss_state = {}
#         if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
#             target_encoding = observations[
#                 IntegratedPointGoalGPSAndCompassSensor.cls_uuid
#             ]
#         elif PointGoalSensor.cls_uuid in observations:
#             target_encoding = observations[PointGoalSensor.cls_uuid]
#         elif ImageGoalSensor.cls_uuid in observations:
#             image_goal = observations[ImageGoalSensor.cls_uuid]
#             target_encoding = self.goal_visual_encoder.foward(
#                 {"rgb": image_goal}
#             )

#         x = [target_encoding]
#         # print("obsevrations", observations["rgb"].shape)
#         if not self.is_blind:
#             perception_embed = self.visual_encoder.forward(observations)
#             x = [perception_embed] + x
#             aux_loss_state["perception_embed"] = perception_embed

#         # x_out = torch.cat(x, dim=1)

#         image_encoding = perception_embed

#         node_feat = self.ring_network.get_node_features(
#             image_encoding, 2, odometry=None
#         )

#         # Set the node and edge features in the graph object
#         for i, feat in enumerate(node_feat):
#             self.ring_network.ran_graph.nodes[i]["feature"] = feat

#         # Generate the input features and edge index tensor for the model
#         x = torch.tensor(node_feat, dtype=torch.float)
#         edge_index = self.ring_network.get_edge_index()
#         edge_index = torch.tensor(edge_index, dtype=torch.long)

#         # Compute the graph embedding using the GCN model
#         embedding = self.state_encoder(x, edge_index)

#         return embedding, aux_loss_state
