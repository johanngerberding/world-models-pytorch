import os
import yaml 
import torch 
import datetime 
import albumentations as A
from torchvision.utils import save_image 

from dataset import ObservationDataset 
from models.vae import VAE


cfg = yaml.safe_load(open("params.yaml"))['vae']
print(f"VAE Training Configuration: {cfg}")
assert cfg['height'] == cfg['width']

def train_epoch(epoch: int) -> None:
    model.train() 
    train_loss = 0 
    train_dataset.load_next_buffer()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        data = torch.transpose(data, 1, 3)
        optimizer.zero_grad()
        reconst, mu, logvar = model(data)
        loss = loss_fn(reconst, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 20 == 0: 
            print(f"train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tloss: {loss.item() / len(data):.6f}")

    print("---> Epoch: {} Average Loss: {:.4f}".format(epoch, train_loss / len(train_loader.dataset)))

def test_epoch() -> float: 
    model.eval()
    test_loss = 0
    test_dataset.load_next_buffer()
    with torch.no_grad(): 
        for data in test_loader:
            data = data.to(device) 
            data = torch.transpose(data, 1, 3)
            reconst, mu, logvar = model(data)
            test_loss += loss_fn(reconst, data, mu, logvar).item()
    
    test_loss /= len(test_loader.dataset)
    print(f"Test loss: {test_loss:.4f}") 
    return test_loss


def loss_fn(reconst, x, mu, logsigma): 
    """VAE loss function"""
    bce = torch.nn.functional.mse_loss(reconst, x, size_average=False)
    kld = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return bce + kld 


device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
td = datetime.datetime.now().strftime("%Y-%m-%d")
exp_dir = f"exps/VAE/{td}_bs_{cfg['train_batch_size']}_epochs_{cfg['num_epochs']}_latentsize_{cfg['latent_size']}"
os.makedirs(exp_dir, exist_ok=False)
weights_dir = os.path.join(exp_dir, "weights")
os.makedirs(weights_dir)
samples_dir = os.path.join(exp_dir, "samples")
os.makedirs(samples_dir)

with open(f'{exp_dir}/params.yaml', 'w') as fp: 
    yaml.dump(cfg, fp, default_flow_style=False)

train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.3), 
        A.Resize(cfg['height'], cfg['width']),  
    ]
) 

test_transform = A.Compose(
    [
        A.Resize(cfg['height'], cfg['width'])
    ]
) 

train_dataset = ObservationDataset(
    root=cfg['data_dir'], 
    transform=train_transform, 
    train=True, 
    buffer_size=100, 
    num_test_files=600,
) 

test_dataset = ObservationDataset(
    root=cfg['data_dir'], 
    transform=test_transform, 
    train=False,
    buffer_size=100, 
    num_test_files=600,
)
print(f"Size of train dataset: {len(train_dataset)}")
print(f"Size of test dataset: {len(test_dataset)}")

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=cfg['test_batch_size'], shuffle=False, num_workers=4
)

model = VAE(latent_size=cfg['latent_size'], in_channels=3)
model = model.to(device)
print("Model created")
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode='min', factor=0.5, patience=5)

best = 1_000_000_000

for epoch in range(1, cfg['num_epochs'] + 1):
    print(f"Training Epoch {epoch}")
    train_epoch(epoch)
    test_loss = test_epoch()
    scheduler.step(test_loss)

    if test_loss < best: 
        best = test_loss
        torch.save(model.state_dict(), os.path.join(weights_dir, "best.pth"))
        print(f"New best model from epoch {epoch}")

    with torch.no_grad(): 
        # this only works if height == width
        sample = torch.randn(cfg['height'], cfg['latent_size']).to(device)
        sample = model.decoder(sample).cpu()
        # this assumes normalization x/255. 
        save_image(sample, os.path.join(samples_dir, f"sample_{epoch}.png"))


torch.save(model.state_dict(), os.path.join(weights_dir, "final.pth"))

