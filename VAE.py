#!/usr/bin/env python3.5

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.nn import functional as F

img_size = 32
latent_size = 5

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        hidden_size = 500
        self.fc1 = nn.Linear(img_size*img_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, img_size*img_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.train:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):

        mu, logvar = self.encode(x.view(-1, img_size*img_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
