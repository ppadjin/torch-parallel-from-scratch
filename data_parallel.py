import os
import subprocess
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

class DataParallel(ABC):
    def __init__(self, gpu_ids):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)

    @abstractmethod
    def train(model, gpu_id):
        """
        This method should be implemented by the subclass, when implementing strategy for data parallelism.
        """
        pass


    @staticmethod
    def get_available_gpus(th=10):
        """
        Avoid using someone else's GPU by checking for available GPUs. Criteria is that I only use GPUs with less than th% utilization.
        """
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        
        )
        assert result.returncode == 0 # check if valid

        available_gpus = []
        for line in result.stdout.strip().split("\n"):
            gpu_index, gpu_utilization = line.split(", ")
            gpu_index = int(gpu_index)
            gpu_utilization = int(gpu_utilization)
            
            if gpu_utilization < th: available_gpus.append(gpu_index)

        return available_gpus


