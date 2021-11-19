import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, filling=False, shuffle=True, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else (dataset_len + batch_size - 1) // batch_size
        self.max_p = self.iters_per_ep * batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.epoch = 0
        self.indices = self.gener_indices()
    
    def gener_indices(self):
        indices = np.arange(self.dataset_len)
        if self.shuffle:
            np.random.shuffle(indices)
        tails = self.batch_size - (self.dataset_len % self.batch_size)
        
        if tails != self.batch_size and self.filling:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))
        
        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())
    
    def __iter__(self):
        self.epoch = 0
        while True:
            self.epoch += 1
            p, q = 0, 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()
    
    def __len__(self):
        return self.iters_per_ep


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(self, world_size, rank, dataset_len, glb_batch_size, filling=False, shuffle=True):
        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size
        
        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.epoch = 0
        self.indices = self.gener_indices()
    
    def gener_indices(self):
        global_max_p = self.iters_per_ep * self.glb_batch_size  # global_max_p % world_size == 0
        # print(f'global_max_p = iters_per_ep({self.iters_per_ep}) * glb_batch_size({self.glb_batch_size}) = {global_max_p}')
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            global_indices = torch.randperm(
                self.dataset_len, generator=g
            )
        else:
            global_indices = torch.arange(self.dataset_len)
        filling = global_max_p - global_indices.shape[0]
        if filling > 0 and self.filling:
            global_indices = torch.cat((global_indices, global_indices[:filling]))
        global_indices = tuple(global_indices.numpy().tolist())
        
        seps = torch.linspace(0, len(global_indices), self.world_size + 1, dtype=torch.int)
        local_indices = global_indices[seps[self.rank]:seps[self.rank + 1]]
        self.max_p = len(local_indices)
        return local_indices
