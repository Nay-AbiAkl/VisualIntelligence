# Ring Attractor Network
We formulate the ring attractor network as a multidirectional graph with N nodes and pass the inputs to the graph as node and edge features. To learn on the graph, we rely on Graph Convolutional Networks.

The graph formulation and GCN network can be found in "ran_graph_gcn.py".

## Approach 1
The first approach consists of using the GCN as a standalone policy. Therefore, we modified the "policy" and "ppo_trainer" files from habitat and created new versions which we refer to as "ran_policy" and "ran_ppo_trainer".

## Approach 2
The second approach consists of integrating the graph and GCN into the existing "PointNavResNetPolicy" in habitat. Slight modifications were added to the ResNet policy, these can be found in "modified_resnet_policy.py".


# ViTMAE