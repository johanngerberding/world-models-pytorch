"""Variational Autoencoder Model""" 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class Encoder(nn.Module): 
    """VAE Encoder""" 
    def __init__(self, latent_size: int, in_channels: int = 3):
        super().__init__()
        self.latent_size = latent_size 

        self.conv1 = nn.Conv2d(in_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2*2*256, self.latent_size)
        self.fc_log_sigma = nn.Linear(2*2*256, self.latent_size)
   

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x)) 
        x = F.relu(self.conv4(x)) 
        x = x.view(x.size(0), -1) 

        mu = self.fc_mu(x) 
        logsigma = self.fc_log_sigma(x)

        return mu, logsigma 


class Decoder(nn.Module): 
    """VAE Decoder""" 
    def __init__(self, latent_size: int, in_channels: int = 3):
        super().__init__()
        self.latent_size = latent_size 
        self.in_channels = in_channels
         
        self.fc = nn.Linear(self.latent_size, 1024) 
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, self.in_channels, 6, stride=2)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconst = F.sigmoid(self.deconv4(x))

        return reconst 


class VAE(nn.Module): 
    """Convolutional Variational Autoencoder""" 
    def __init__(self, latent_size: int, in_channels: int = 3):
        super().__init__()
        self.latent_size = latent_size
        self.in_channels = in_channels 
        self.encoder = Encoder(self.latent_size, self.in_channels)
        self.decoder = Decoder(self.latent_size, self.in_channels)

    def forward(self, x) -> tuple:
        mu, logsigma = self.encoder(x) 
        sigma = logsigma.exp() 
        # N(0,1) -> sample from normal distribution with mean 0, variance 1 
        eps = torch.randn_like(sigma)
        # z = mu + sigma * N(0,1)
        z = eps.mul(sigma).add_(mu) 
        reconst = self.decoder(z)
        return reconst, mu, logsigma 

