import torch 
import albumentations as A 

from dataset import RolloutDataset
from model import VAE

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
    BCE = torch.nn.functional.mse_loss(reconst, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD

data_dir = "/home/mojo/dev/world-models-pytorch/data" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10
latent_size = 512
train_batch_size = 256 
test_batch_size = 128 
learning_rate = 0.01
height = 64 
width = 64

train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.3), 
        A.Resize(height, width),  
    ]
) 

test_transform = A.Compose(
    [
        A.Resize(height, width)
    ]
) 

train_dataset = RolloutDataset(
    root=data_dir, 
    transform=train_transform, 
    train=True, 
    buffer_size=100, 
    num_test_files=600,
) 

test_dataset = RolloutDataset(
    root=data_dir, 
    transform=test_transform, 
    train=False,
    buffer_size=100, 
    num_test_files=600,
)
print(f"Size of train dataset: {len(train_dataset)}")
print(f"Size of test dataset: {len(test_dataset)}")

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4
)

model = VAE(latent_size=latent_size, in_channels=3)
model = model.to(device)
print("Model created")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode='min', factor=0.5, patience=5)

for epoch in range(1, num_epochs + 1):
    print(f"Training Epoch {epoch}")
    train_epoch(epoch)
    test_loss = test_epoch()

