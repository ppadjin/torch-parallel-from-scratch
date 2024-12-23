import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import time 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import numpy as np
import random
import torch.nn as nn
from timeit import timeit
import matplotlib.pyplot as plt
from src.model import MobileNetV2Like
import wandb

from src.data_parallel import RingAllReduce, ParameterServer
from src.datamanager import DataManager
from src.utils import get_data



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--strategy", type=str, default="ring_all_reduce", choices=["ring_all_reduce", "param_server"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--train_size", type=int, default=1000)
    args = parser.parse_args()

    # Instantiate the model
    model = MobileNetV2Like(input_shape=(3, 32, 32), num_classes=10)

    # Load CIFAR-10 dataset
    train_subset = get_data(train_size=args.train_size)
    
    if args.use_wandb:
        wandb.login()
        wandb_config = {
            "n_gpus": args.n_gpus,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "strategy": args.strategy
        }
        wandb_project_name = 'mobilenet'
        wandb_setup = {
            "project": wandb_project_name,
            "config": wandb_config
        }
        

    if args.strategy == "ring_all_reduce":
        datamanager = DataManager(train_subset, args.n_gpus, strategy=args.strategy, batch_size=args.batch_size)
        ring_all_reduce = RingAllReduce(datamanager, num_gpus=args.n_gpus, use_wandb=args.use_wandb, wandb_setup=wandb_setup)
        ring_all_reduce.train(model, epochs=args.epochs, lr=args.lr)

    elif args.strategy == "param_server":
        datamanager = DataManager(train_subset, args.n_gpus, strategy=args.strategy, batch_size=args.batch_size)
        param_server = ParameterServer(datamanager, num_gpus=args.n_gpus, use_wandb=args.use_wandb, wandb_setup=wandb_setup)
        param_server.train(model, epochs=args.epochs, lr=args.lr)