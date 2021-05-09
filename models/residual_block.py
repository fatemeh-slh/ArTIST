from __future__ import print_function
import torch
from torch import nn


class ResidualBlock(nn.Module):
    """ Residual Network that is then used for the VAE encoder and the VAE decoder. """

    def __init__(self, input_size, embedding_size):
        super().__init__()

        self.shortcut = nn.Linear(input_size, embedding_size)
        self.deep1 = nn.Linear(input_size, embedding_size // 2)
        self.deep2 = nn.Linear(embedding_size // 2, embedding_size // 2)
        self.deep3 = nn.Linear(embedding_size // 2, embedding_size)

    def forward(self, input_tensor, activation=nn.ReLU()):

        if activation is not None:

            shortcut = self.shortcut(input_tensor)

            deep1 = activation(self.deep1(input_tensor))
            deep2 = activation(self.deep2(deep1))
            deep3 = self.deep3(deep2)

        else:

            shortcut = self.shortcut(input_tensor)

            deep1 = self.deep1(input_tensor)
            deep2 = self.deep2(deep1)
            deep3 = self.deep3(deep2)

        output = shortcut + deep3

        return output
