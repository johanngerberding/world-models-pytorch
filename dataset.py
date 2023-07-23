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
        return self._get_data(data, seq_idx)

    def _get_data(self, data, idx): 
        pass 

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


class ObservationDataset(RolloutDataset): 
    def _get_data(self, data, idx: int):
        obs = data['observations'][idx] 
        if self.transform: 
            transformed = self.transform(image=obs)
            obs = transformed['image']
        obs = torch.tensor(obs).float()
        obs = obs / 255.
        return obs


class SequenceDataset(RolloutDataset): 
    def __init__(
            self, 
            root: str, 
            transform: Compose, 
            train: bool, 
            buffer_size: int, 
            num_test_files: int, 
            seq_len: int
        ):
        super().__init__(root, transform, train, buffer_size, num_test_files) 
        self.seq_len = seq_len
    
    def _get_data(self, data, idx: int):
        obs_data = data['observations'][idx:idx + self.seq_len + 1]
        if self.transform: 
            transformed_obs = self.transform(image=obs_data.astype(np.float32))
            obs_data = transformed_obs['image']
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data['actions'][idx:idx + self.seq_len + 1]
        action = action.astype(np.float32)
        reward = data['rewards'][idx+1:idx + self.seq_len + 1].astype(np.float32)
        terminal = data['terminals'][idx+1:idx + self.seq_len + 1].astype(np.float32)
        
        return obs, action, reward, terminal, next_obs 



if __name__ == "__main__":
    dataset = SequenceDataset("/home/mojo/dev/world-models-pytorch/data", None, True, 100, 600, 100)
    print(len(dataset))
    from torch.utils.data import DataLoader 
    dataloader = DataLoader(dataset, 4, shuffle=True)

    for data in dataloader: 
        print(len(data))
        obs, action, reward, terminal, next_obs = data 
        print(obs.shape) 
        print(action.shape)
        break 
