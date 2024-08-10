import os 
import argparse 
import torch 
from torch.utils.data import DataLoader 

from models.rnn import MDRNN
from models.vae import VAE

from dataset import SequenceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len = 32 
batch_size = 16 
num_epochs = 25
latent_size = 64
num_gaussians = 5  # from the paper 
hidden_size = 256
action_size = 3 
learning_rate = 0.0001

# load vae model
vae_dir = "/home/mojo/dev/world-models-pytorch/exps/VAE/2023-07-23_bs_512_epochs_50_latentsize_64"
vae_weights = os.path.join(vae_dir, "weights", "best.pth")
model = VAE(latent_size=latent_size, in_channels=3) 
model.load_state_dict(torch.load(vae_weights)) 
print(model)
print("Loaded VAE.")
model.to(device)

# create rnn 
mdrnn = MDRNN(latents=latent_size, actions=action_size, hiddens=hidden_size, gaussians=num_gaussians)
mdrnn.to(device)

optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=learning_rate, alpha=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode='min', factor=0.5, patience=5) 

transform = None 
train_dataset = SequenceDataset(
    root="/data/world-models", 
    transform=transform, 
    train=True, 
    buffer_size=100, 
    num_test_files=600, 
    seq_len=seq_len,
)
print(f"Len Train dataset: {len(train_dataset)}") 

test_dataset = SequenceDataset(
    root="/data/world-models", 
    transform=transform,
    train=False, 
    buffer_size=100, 
    num_test_files=600,
    seq_len=seq_len,
)
print(f"Len Test dataset: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)


for data in train_dataloader: 
    print(len(data))
    obs, action, reward, terminal, next_obs = data 
    print(f"obs shape: {obs.shape}")
    print(f"action shape: {action.shape}")
    print(f"reward shape: {reward.shape}")
    print(f"terminal shape: {terminal.shape}")
    print(f"next obs shape: {next_obs.shape}")

    break 