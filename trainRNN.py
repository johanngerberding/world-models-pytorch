import os
import yaml
import torch 
import shutil
from torch.utils.data import DataLoader 
from datetime import datetime
import torch.nn.functional as F
from models.rnn import MDRNN
from models.vae import VAE
from losses import gmm_loss
from dataset import SequenceDataset


cfg = yaml.safe_load(open("params.yaml"))
rnn_cfg = cfg['rnn']
print(f"MDN-RNN Training Configuration: {rnn_cfg}")

device = torch.device(rnn_cfg['device'] if torch.cuda.is_available() else "cpu")

# load vae model
vae_weights = os.path.join(rnn_cfg['vae_dir'], "weights", "best.pth")
vae = VAE(latent_size=rnn_cfg['latent_size'], in_channels=3) 
vae.load_state_dict(torch.load(vae_weights)) 
print(vae)
print("Loaded VAE.")
vae.to(device)

# create rnn 
mdrnn = MDRNN(
    latents=rnn_cfg['latent_size'], 
    actions=rnn_cfg['action_size'], 
    hiddens=rnn_cfg['hidden_size'], 
    gaussians=rnn_cfg['num_gaussians'],
)
mdrnn.to(device)

mdrnn_dir = f'exps/MDRNN/{datetime.now().strftime("%Y-%m-%d")}'
if os.path.exists(mdrnn_dir):
    shutil.rmtree(mdrnn_dir)    
os.makedirs(mdrnn_dir)

optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=rnn_cfg['learning_rate'], alpha=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode='min', factor=0.5, patience=5) 

transform = None 
train_dataset = SequenceDataset(
    root="/data/world-models", 
    transform=transform, 
    train=True, 
    buffer_size=30, 
    num_test_files=600, 
    seq_len=rnn_cfg['seq_len'],
)
print(f"Len Train dataset: {len(train_dataset)}") 

test_dataset = SequenceDataset(
    root="/data/world-models", 
    transform=transform,
    train=False, 
    buffer_size=10, 
    num_test_files=600,
    seq_len=rnn_cfg['seq_len'],
)
print(f"Len Test dataset: {len(test_dataset)}")

# maybe this is needed because of some bugs while training
def collate_fn(batch): 
    ...


train_dataloader = DataLoader(train_dataset, batch_size=rnn_cfg['batch_size'], num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=rnn_cfg['batch_size'], num_workers=8)

def transform_to_latent(obs, model) -> torch.Tensor:
    obs = obs / 255. 
    obs = [torch.nn.functional.upsample(
        frame.view(-1, 3, 96, 96), 
        size=64, 
        mode='bilinear', 
        align_corners=True,
    ) for frame in obs]

    mus, sigmas = [], []
    for frame in obs: 
        _, mu, sigma = model(frame.to(device))
        mus.append(mu) 
        sigmas.append(sigma)
    
    latents = [] 
    for mu, sigma in zip(mus, sigmas):
        latent = mu + sigma.exp() * torch.randn_like(mu)
        latents.append(latent) 
    
    latents = torch.stack(latents)
    latents = latents.view(rnn_cfg['batch_size'], rnn_cfg['seq_len'], rnn_cfg['latent_size']) 

    return latents 

def step(dataloader, rnn, vae, optimizer, train: bool) -> float:
    if train: 
        rnn.train()
    else: 
        rnn.eval()
    
    cum_loss = 0
    for data in dataloader: 
        obs, action, reward, terminal, next_obs = data 
        obs = obs.to(device)
        action = action.to(device)
        next_obs = next_obs.to(device)
        reward = reward.to(device)
        terminal = terminal.to(device)
        print(obs.shape)
        # use VAE to turn observation and next_observation to latent 
        with torch.no_grad():  
            latent = transform_to_latent(obs=obs, model=vae) 
            next_obs_latent = transform_to_latent(obs=next_obs, model=vae) 
        
        # prep data stuff
        latent = latent.transpose(1, 0)
        next_obs_latent = next_obs_latent.transpose(1, 0)

        obs = obs.transpose(1, 0)
        action = action.transpose(1, 0)
        # add the action dimension
        action = F.one_hot(action.to(torch.int64), num_classes=5)
        next_obs = next_obs.transpose(1, 0)
        reward = reward.transpose(1, 0)
        terminal = terminal.transpose(1, 0)

        # mdrnn forward pass 
        mus, sigmas, logpi, rewards, dones = rnn(action, latent)
        # calculate losses 
        gmm = gmm_loss(next_obs_latent, mus, sigmas, logpi) 
        bce = torch.nn.functional.binary_cross_entropy_with_logits(dones, terminal)

        if rnn_cfg['include_reward']: 
            mse = torch.nn.functional.mse_loss(rewards, reward)
            scale = rnn_cfg['seq_len'] + 2
        else: 
            mse = 0
            scale = rnn_cfg['seq_len'] + 1 

        loss = (gmm + bce + mse) / scale
        print(f"Loss: {loss}")
        cum_loss += loss        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return cum_loss * (rnn_cfg['batch_size']) / len(dataloader.dataset)


cur_best = None 

for e in range(rnn_cfg['num_epochs']):
    step(train_dataloader, mdrnn, vae, optimizer, True)
    test_loss = step(test_dataloader, mdrnn, vae, optimizer, False)

    is_best = not cur_best or test_loss < cur_best
    if is_best: 
        cur_best = is_best
        print(f"New best model, loss: {cur_best}")

