import torch 
import torch.nn as nn 
import torch.nn.functional as f 


class MDRNN(nn.Module):
    """Model for multi-step forward"""
    def __init__(self, latents: int, actions: int, hiddens: int, gaussians: int):
        super().__init__()
        self.latents = latents # z  
        self.actions = actions # a  
        self.hiddens = hiddens # h  
        self.gaussians = gaussians # number of gaussians

        self.rnn = nn.LSTM(
            input_size=latents + actions, 
            hidden_size=hiddens,
            num_layers=1,
            bias=True, 
            dropout=0,
            bidirectional=False 
        )
        self.gmm = nn.Linear(
            in_features=hiddens,
            out_features=(2 * latents + 1) * gaussians + 2
        )

    def forward(self, actions: torch.Tensor, latents: torch.Tensor) -> tuple:
        """This is for the whole sequence [training]""" 
        seq_len, batch_size = actions.size(0), actions.size(1)
        inputs = torch.cat([actions, latents], dim=-1) 
        outs, _ = self.rnn(inputs)
        mdn_outs = self.gmm(outs)

        stride = self.gaussians * self.latents  
        mus = mdn_outs[:, :, :stride]
        mus = mus.view(seq_len, batch_size, self.gaussians, self.latents)

        sigmas = mdn_outs[:, :, stride:stride*2]
        sigmas = sigmas.view(seq_len, batch_size, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = mdn_outs[:, :, 2*stride:2*stride + self.gaussians]
        pi = pi.view(seq_len, batch_size, self.gaussians)
        # for the weighting for each gaussian 
        logpi = torch.nn.functional.log_softmax(pi, dim=-1)

        rewards = mdn_outs[:, :, -2]  # rewards
        dones = mdn_outs[:, :, -1]  # dones

        return mus, sigmas, logpi, rewards, dones 
    

class MRDNNCell(nn.Module):
    """Model for one step forward (training the controller)"""
    def __init__(self, latents: int, actions: int, hiddens: int, gaussians: int):
        super().__init__() 
        self.latents = latents 
        self.actions = actions 
        self.hiddens = hiddens 
        self.gaussians = gaussians
        self.rnn = nn.LSTMCell(latents + actions, hiddens)
        self.gmm = nn.Linear(
            in_features=hiddens,
            out_features=(2 * latents + 1) * gaussians + 2
        )

    def forward(self, action: torch.Tensor, latent: torch.Tensor, hidden: torch.Tensor) -> tuple:
        """one step forward""" 
        inputs = torch.cat([action, latent], dim=1) 
        next_hidden = self.rnn(inputs, hidden)
        out_rnn = next_hidden[0]
        out_full = self.gmm(out_rnn)

        stride = self.gaussians + self.latents
        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride: 2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        rewards = out_full[:, -2]
        dones = out_full[:, -1]

        return mus, sigmas, logpi, rewards, dones, next_hidden

