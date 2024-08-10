import os 
import yaml
import torch 
import shutil
from torch.utils.data import DataLoader 
from datetime import datetime
from models.rnn import MDRNN
from models.vae import VAE

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
    buffer_size=100, 
    num_test_files=600, 
    seq_len=rnn_cfg['seq_len'],
)
print(f"Len Train dataset: {len(train_dataset)}") 

test_dataset = SequenceDataset(
    root="/data/world-models", 
    transform=transform,
    train=False, 
    buffer_size=100, 
    num_test_files=600,
    seq_len=rnn_cfg['seq_len'],
)
print(f"Len Test dataset: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=rnn_cfg['batch_size'], num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=rnn_cfg['batch_size'], num_workers=8)

c = 0 
for data in train_dataloader: 
    c += 1  
    obs, action, reward, terminal, next_obs = data 

    print(f"obs shape: {obs.shape}")
    print(f"action shape: {action.shape}")
    print(f"reward shape: {reward.shape}")
    print(f"terminal shape: {terminal.shape}")
    print(f"next obs shape: {next_obs.shape}")

    # use VAE to turn observation to latent 
    with torch.no_grad(): 
        # vae input -> bs * seq_len, channels, img_width, img_height 
        vae_obs = obs.view(-1, 3, cfg['vae']['width'], cfg['vae']['height']) 
        print(vae_obs.size()) 
        vae_obs = torch.nn.functional.upsample(vae_obs, size=64, mode='bilinear', align_corners=True) 
        print(vae_obs.size()) 
        # reconst, mu, logsigma = vae()

    # 
    if c == 3:
        break 