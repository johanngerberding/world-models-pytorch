import os
import yaml
import torch 
import tqdm
import shutil
from torch.utils.data import DataLoader 
from datetime import datetime
import torch.nn.functional as F
from models.rnn import MDRNN
from models.vae import VAE
from losses import gmm_loss
from dataset import SequenceDataset

from utils import collate_fn, transform_to_latent


cfg = yaml.safe_load(open("params.yaml"))
rnn_cfg = cfg['rnn']
print(f"MDN-RNN Training Configuration: \n{rnn_cfg}")

device = torch.device(rnn_cfg['device'] if torch.cuda.is_available() else "cpu")

# load vae model
vae_weights = os.path.join(rnn_cfg['vae_dir'], "weights", "best.pth")
vae = VAE(latent_size=rnn_cfg['latent_size'], in_channels=3) 
vae.load_state_dict(torch.load(vae_weights)) 
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
    buffer_size=rnn_cfg['buffer_size'], 
    num_test_files=600, 
    seq_len=rnn_cfg['seq_len'],
)
print(f"Len Train dataset: {len(train_dataset.files)}") 

test_dataset = SequenceDataset(
    root="/data/world-models", 
    transform=transform,
    train=False, 
    buffer_size=rnn_cfg['buffer_size'], 
    num_test_files=600,
    seq_len=rnn_cfg['seq_len'],
)
print(f"Len Test dataset: {len(test_dataset.files)}")

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=rnn_cfg['batch_size'], 
    num_workers=rnn_cfg['train_num_workers'],
    collate_fn=collate_fn,
    drop_last=True,
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=rnn_cfg['batch_size'], 
    num_workers=rnn_cfg['test_num_workers'], 
    collate_fn=collate_fn,
    drop_last=True,
)


def step(dataloader, rnn, vae, optimizer, train: bool, epoch: int) -> float:
    if train: 
        rnn.train()
    else: 
        rnn.eval()

    dataloader.dataset.load_next_buffer()
    
    cum_loss = 0
    cum_gmm = 0 
    cum_mse = 0 
    cum_bce = 0 

    pbar = tqdm.tqdm(total=len(dataloader.dataset), desc=f"Epoch {epoch + 1} ")
    for i, data in enumerate(dataloader): 
        obs, action, reward, terminal, next_obs = data 
        obs = obs.to(device)
        action = action.to(device)
        next_obs = next_obs.to(device)
        reward = reward.to(device)
        terminal = terminal.to(device)

        # at the end of the rollout you will have shorter seq_len than 100
        # this makes the training work, but it is a bit hacky and I think there is 
        # a better way of doing it 
        # when the seq-len gets shorter than 20 we load the next buffer 
        if obs.shape[1] <= 20: 
            dataloader.dataset.load_next_buffer()
            pbar.update(rnn_cfg['batch_size'])
            continue

        # use VAE to turn observation and next_observation to latent 
        with torch.no_grad():  
            latent = transform_to_latent(obs=obs, model=vae, device=device, rnn_cfg=rnn_cfg) 
            next_obs_latent = transform_to_latent(obs=next_obs, model=vae, device=device, rnn_cfg=rnn_cfg) 
        
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
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        cum_loss += loss.item()        
        cum_gmm += gmm.item()
        cum_mse += mse.item() 
        cum_bce += bce.item() 

        pbar.set_postfix_str(
            "loss={loss:5.4f} | bce={bce:5.4f} | gmm={gmm:5.4f} | mse={mse:5.4f}".format(
                loss=cum_loss / (i + 1), bce=cum_bce / (1 + i), gmm=cum_gmm / rnn_cfg['latent_size'] / (1 + i), mse=cum_mse / (i + 1)
            )
        )
        pbar.update(rnn_cfg["batch_size"])

    pbar.close()
    # not sure if that is correct 
    return cum_loss * (rnn_cfg['batch_size']) / len(dataloader.dataset)


cur_best = None 

for e in range(rnn_cfg['num_epochs']):
    step(train_dataloader, mdrnn, vae, optimizer, True, e)
    test_loss = step(test_dataloader, mdrnn, vae, optimizer, False, e)
    scheduler.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best: 
        cur_best = is_best
        print(f"New best model, loss: {test_loss:10.6f}")

    checkpoint_path = os.path.join(mdrnn_dir, f"checkpoint-epoch-{str(e + 1).zfill(3)}.pth")
    torch.save(
        {
            'epoch': e, 
            'model_state_dict': mdrnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss,
        }, checkpoint_path
    )