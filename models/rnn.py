import torch.nn as nn 


class MDRNN(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents 
        self.actions = actions 
        self.hiddens = hiddens 
        self.gaussians = gaussians

    def forward(self, actions, latents):
        pass