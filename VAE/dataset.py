import glob
import torch 
import numpy as np 
from torch.utils.data import Dataset 
from albumentations.core.composition import Compose
from bisect import bisect

class RolloutDataset(Dataset):
    def __init__(
        self, 
        root: str, 
        transform: Compose, 
        train: bool = True, 
        buffer_size: int = 100, 
        num_test_files: int = 600
    ):
        super().__init__() 
        self.root = root 
        self.transform = transform
        self.files = glob.glob(self.root + "/**/*.npz", recursive=True)
        if train:
            self.files = self.files[:-num_test_files]
        else: 
            self.files = self.files[-num_test_files:]
        
        self.cum_size = None 
        self.buffer = None 
        self.buffer_size = buffer_size 
        self.buffer_idx = 0 
        self.buffer_fnames = None 

    def __len__(self):
        if not self.cum_size: 
            self.load_next_buffer() 
        return self.cum_size[-1]

    def __getitem__(self, idx: int):
        # binary search into cum_size to find the file where to look 
        file_idx = bisect(self.cum_size, idx) - 1
        seq_idx = idx - self.cum_size[file_idx] 
        data = self.buffer[file_idx] 
        obs = data['observations'][seq_idx]
        # using albumentation for transforms 
        if self.transform: 
            transformed = self.transform(image=obs)
            obs = transformed['image']
        obs = torch.tensor(obs).float()
        obs = obs / 255.
        return obs  

    def load_next_buffer(self):
        """Load next buffer, dataset is too big to store in ram""" 
        self.buffer_fnames = self.files[self.buffer_idx:self.buffer_idx + self.buffer_size]
        self.buffer_idx += self.buffer_size
        self.buffer_idx = self.buffer_idx % len(self.files)
        self.buffer = [] 
        self.cum_size = [0]
        
        for f in self.buffer_fnames: 
            with np.load(f) as data: 
                self.buffer += [{k: np.copy(v) for k, v in data.items()}]
                self.cum_size += [self.cum_size[-1] + data['rewards'].shape[0]]

            

if __name__ == "__main__":
    dataset = RolloutDataset("/home/mojo/dev/world-models-pytorch/data", None, True, 100, 600)
    print(len(dataset))
    from torch.utils.data import DataLoader 
    dataloader = DataLoader(dataset, 4, shuffle=True)

    for obs in dataloader: 
        print(obs.shape)
        break 
