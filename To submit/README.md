# Introduction 

This repo contains all scripts and notebooks used for the project titled "Bio-inspired visually guided neuromorphic robotic navigation" for CS-503 course at EPFL. Inspired from the ring-attractor network, we build a GCN-based ring attractor network and train it within an RL framework. The scripts included in this repo are to be substituted directly within the Habitat lab package files for our experiments to be reproduced. The main notebook titled "ran_notebook.ipynb" acts as the main script we used to run our experiments. 

# Ring Attractor Network
We formulate the ring attractor network as a multidirectional graph with N nodes and pass the inputs to the graph as node and edge features. To learn on the graph, we rely on Graph Convolutional Networks.

The graph formulation and GCN network can be found in "ran_graph_gcn.py".

## Approach 1
The first approach consists of using the GCN as a standalone policy. Therefore, we modified the "policy" and "ppo_trainer" files from habitat and created new versions which we refer to as "ran_policy" and "ran_ppo_trainer".

## Approach 2
The second approach consists of integrating the graph and GCN into the existing "PointNavResNetPolicy" in habitat. Slight modifications were added to the ResNet policy, these can be found in "modified_resnet_policy.py".

This approach was the one we presented. All experiments conducted within this approach are displayed in the "ran_notebook.ipynb"; you can find all the corresponding tensorboard files in the "experiment_tensorboards" folder.
# ViTMAE
The ViTMAE was finetuned on examples from gibson environment using the notebook titles "vitmae.ipynb". The notebook shows the different functions used for finetuning and the metrics used for assessment which are MSE, SSIM and PSNR. This notebook, specially the visualisation, are based on the [HuggingFace's notebook demo](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb) . Moreover, the dataset used for finetuning could be found in [this file](https://drive.google.com/drive/folders/1hetIgzuA4mhH08Udn2vef0Lt48NdAve7?usp=sharing)
