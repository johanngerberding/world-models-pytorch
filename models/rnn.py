import torch 
import torch.nn as nn 


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
        self.mdn = nn.Linear(
            in_features=hiddens,
            out_features=(2 * latents + 1) * gaussians + 2
        )

    def forward(self, actions: torch.Tensor, latents: torch.Tensor) -> tuple:
        """This is for the whole sequence [training]""" 
        seq_len, batch_size = actions.size(0), actions.size(1)
        inputs = torch.cat([actions, latents], dim=-1) 
        outs, _ = self.rnn(inputs)
        mdn_outs = self.mdn(outs)

        stride = self.gaussians * self.latents  
        mus = mdn_outs[:, :, :stride]
        mus = mus.view(seq_len, batch_size, self.gaussians, self.latents)

        sigmas = mdn_outs[:, :, stride:stride*2]
        sigmas = sigmas.view(seq_len, batch_size, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = mdn_outs[:, :, 2*stride:2*stride + self.gaussians]
        pi = pi.view(seq_len, batch_size, self.gaussians)
        logpi = torch.nn.functional.log_softmax(pi, dim=-1)

        rs = mdn_outs[:, :, -2]
        ds = mdn_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds 
    

class MRDNNCell(nn.Module):
    """Model for one step forward"""
    def __init__(self, latents: int, actions: int, hiddens: int, gaussians: int):
        super().__init__() 
        self.rnn = nn.LSTMCell(latents + actions, hiddens)
    
    def forward(self, action, latent, hidden):
        pass
