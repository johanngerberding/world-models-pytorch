import glob
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
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
            print("Load new buffer")
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

    def _data_per_seqence(self, data_length): 
        return data_length

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
                self.cum_size += [self.cum_size[-1] + self._data_per_seqence(data['rewards'].shape[0])]


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
        print(f"Index: {idx}")
        obs_data = data['observations'][idx:idx + self.seq_len + 1]
        if self.transform: 
            transformed_obs = self.transform(image=obs_data.astype(np.float32))
            obs_data = transformed_obs['image']
        obs, next_obs = obs_data[:-1], obs_data[1:]
        print(f"obs shape: {obs.shape}") 
        print(f"next obs shape: {next_obs.shape}")
        if obs.shape[0] != 100: 
            print("PROBLEEEEEEM")
            raise Exception
        if next_obs.shape[0] != 100: 
            print("PROBLEEEEEEM")
            raise Exception
        action = data['actions'][idx+1:idx + self.seq_len + 1]
        action = action.astype(np.float32)
        print(f"action: {action.shape}") 
        reward = data['rewards'][idx+1:idx + self.seq_len + 1].astype(np.float32)
        print(f"reward: {reward.shape}") 
        terminal = data['terminals'][idx+1:idx + self.seq_len + 1].astype(np.float32)
        print(f"terminal: {terminal.shape}") 
        return obs, action, reward, terminal, next_obs 
    
    def _data_per_sequence(self, data_length): 
        return data_length - self.seq_len

def collate_fn(batch): 
    # print(f"batch: {len(batch)}")
    obss = [] 
    actions = [] 
    rewards = []   
    terminals = []
    next_obss = [] 
    for sample in batch: 
        obs, action, reward, terminal, next_obs = sample 
        obss.append(torch.tensor(obs))
        actions.append(torch.tensor(action))
        rewards.append(torch.tensor(reward))
        terminals.append(torch.tensor(terminal))
        next_obss.append(torch.tensor(next_obs))

    # print(f"obs: {obss[8].shape}")
    # print(f"actions: {actions[8].shape}")
    # print(f"rewards: {rewards[8].shape}")
    # print(f"terminal: {terminals[8].shape}")
    # print(f"next obs: {next_obss[8].shape}")
    # print(f"terminals: {terminals[8]}")

    obss = torch.stack(obss)
    actions = torch.stack(actions) 
    rewards = torch.stack(rewards)
    terminals = torch.stack(terminals)
    next_obss = torch.stack(next_obss)

    # print(f"obs: {obss.shape}")
    # print(f"actions: {actions.shape}")
    # print(f"rewards: {rewards.shape}")
    # print(f"terminal: {terminals.shape}")
    # print(terminal)
    # print(f"next obs: {next_obss.shape}")
    
    return obss, actions, rewards, terminals, next_obss

def main(): 
    
    train_dataset = SequenceDataset(
        root="/data/world-models", 
        transform=None, 
        train=True, 
        buffer_size=16, 
        num_test_files=600, 
        seq_len=100,
    ) 
    print(f"Len Train dataset: {len(train_dataset.files)}") 
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=16, 
        num_workers=8,
        collate_fn=collate_fn,
    )   
    import tqdm
    
    for data in tqdm.tqdm(train_dataloader): 
        obs, action, reward, terminal, next_obs = data 


    
    # data_files = glob.glob("/data/world-models/**/*.npz", recursive=True)
    # print(f"number of files: {len(data_files)}")
    # problems = {} 
    # for f in tqdm.tqdm(data_files): 
    #     with np.load(f) as data: 
    #         d = {k: np.copy(v) for k, v in data.items()}
    #         l = d['observations'].shape[0]
            
    #     if l != 1000: 
    #         print(f"problem with :{f} - length: {l}") 
    #         problems[f] = l

    # print(problems)


if __name__ == "__main__": 
    main()
