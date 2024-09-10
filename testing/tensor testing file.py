import torch 
from PIL import Image
import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import matplotlib
latent_dim = 3
test = torch.Tensor([[1,1,1],
                     [1,1,1],
                     [1,1,1]])

c = torch.zeros(latent_dim, latent_dim)
c[torch.tril_indices(latent_dim, latent_dim, offset=-1).tolist()] = test[-1:]
c

print(c)

for i in range(config.correlation_linears):    
            self.layers.append(nn.Linear(init_channels*(2**config.correlation_convolutions),
                                         init_channels*(2**config.correlation_convolutions)))
            self.layers.append(nn.LeakyReLU())

        # makes sure that the output of the correlation linear layers is equal to the latent dimensions squared so that
        # there is enough values to make a correlation matrix

        self.layers.append(nn.Linear(init_channels*(2**config.correlation_convolutions),latent_dim**2))

        self.cor_interpretation = torch.nn.Sequential(*self.layers)

for i in range(config.encoder_linears):
                self.layers.append(nn.Linear(init_channels*(2**config.encoder_convolutions),
                                            init_channels*(2**config.encoder_convolutions)))
                self.layers.append(nn.LeakyReLU())
        
        self.latent_interpretation =torch.nn.Sequential(*self.layers)