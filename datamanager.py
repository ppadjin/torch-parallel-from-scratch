import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod

import torch.utils.data.dataset
from utils import divide_to_chunks
from typing import Union


class DataManager:
    def __init__(self, dataset: Union[torch.Tensor, torch.utils.data.Dataset], n_gpus: int, strategy: str, batch_size: int = 32):
        
        self.batch_size = batch_size
        if strategy == 'ring_all_reduce':
            self.n_split = n_gpus
        elif strategy == 'param_server':
            self.n_split = n_gpus - 1

        self.n = len(dataset)
        self.chunk_size = self.n // self.n_split

        if isinstance(dataset, torch.Tensor):
            self.dataset = torch.utils.data.TensorDataset(dataset)
        elif isinstance(dataset, torch.utils.data.Dataset):
            self.dataset = dataset

        split_chunks = [self.chunk_size] * self.n_split
        for i in range(self.n % self.n_split): # fill it up to add up to the length of dataset
            split_chunks[i] += 1

        self.datasets = torch.utils.data.random_split(self.dataset, split_chunks)

        if strategy == 'param_server':
            self.datasets = [dataset, *self.datasets] # make it so that param server has the full dataset, so it can evaluate the accuracy of the model
        
        self.dataloaders = [DataLoader(d, batch_size=self.batch_size, shuffle=True) for d in self.datasets]

    def get_dataloader(self, worker_id):
        return self.dataloaders[worker_id]      
