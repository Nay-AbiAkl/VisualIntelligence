import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class NNeuronLayer(nn.Module):
    def __init__(self, num_neurons):
        super(NNeuronLayer, self).__init__()

        # Define the learnable weight matrices for the self-connections
        # and interconnections
        self.self_weight = nn.Parameter(torch.randn(num_neurons))
        self.inter_weight = nn.Parameter(torch.randn(num_neurons, num_neurons))

        # Initialize the activity of each neuron to zero
        self.activity = nn.Parameter(torch.zeros(num_neurons))

    def forward(self, input):
        # Calculate the net input to each neuron based on the activity of itself
        # and the other neurons
        net_input = []
        for i in range(len(input)):
            net_i = input[i] * self.self_weight[i]
            for j in range(len(input)):
                if j != i:
                    net_i += self.activity[j] * self.inter_weight[i, j]
            net_input.append(net_i)

        # Update the activity of each neuron using a sigmoid activation function
        gain = 2

        self.activity = nn.Parameter(
            1 / (1 + torch.exp(-gain * torch.stack(net_input)))
        )
        # self.activity = net_input

        # Return the activity of each neuron
        return self.activity


# initialize the network
RAN = NNeuronLayer(3)

for param in RAN.parameters():
    print(type(param), param.size())


device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = torch.optim.Adam(params=RAN.parameters(), lr=0.0001)
loss_func = torch.nn.CrossEntropyLoss().to(device)


epochs = 15
steps = 0
print_every = 10
input = [torch.tensor([1, 0, 0]), torch.tensor([0, 1, 0]), torch.tensor([0, 0, 1])]
labels = torch.tensor([0, 1, 2])

epochs = 100

for e in range(epochs):
    start = time.time()
    # move the input and label tensors to the GPU if available
    steps += 1

    # zero the parameter gradients
    optimizer.zero_grad()

    # do the forward pass
    output = RAN(input)
    print("output", output)

    # calculate the loss based on the criterion loss function
    loss = loss_func(output, labels)
    print("loss", loss)

    # backpropagate the loss
    loss.backward()

    # optimize the parameters
    optimizer.step()

    if steps % print_every == 0:
        stop = time.time()
        # Test accuracy
        correct = 0
        total = 1

        # get the predicted class from the maximum value in the output-list
        # of class scores
        correct += (output == labels).sum().item()

        print(
            "Accuracy of the network on the %d val images: \
            %d %%"
            % (total, 100 * correct / total)
        )

        start = time.time()
