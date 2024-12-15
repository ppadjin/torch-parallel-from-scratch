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
    def __init__(self, dataset: Union[torch.Tensor, torch.utils.data.Dataset], n_split: int, batch_size: int = 32):
        self.batch_size = batch_size
        self.n_split = n_split
        self.n = len(dataset)
        self.chunk_size = self.n // n_split

        if isinstance(dataset, torch.Tensor):
            self.dataset = torch.utils.data.TensorDataset(dataset)
        elif isinstance(dataset, torch.utils.data.Dataset):
            self.dataset = dataset

        self.datasets = torch.utils.data.random_split(self.dataset, [self.chunk_size] * n_split)
        
        self.dataloaders = [DataLoader(dataset, batch_size=self.batch_size, shuffle=True) for dataset in self.datasets]

    def get_dataloader(self, worker_id):
        return self.dataloaders[worker_id]      
