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
import torchsummary
import wandb

from data_parallel import RingAllReduce, ParameterServer
from datamanager import DataManager

class InvertedResidualBlock(nn.Module):
    """
    Class taken from MLSys Lab 2
    """
    def __init__(self, in_channels, filters, alpha, stride, expansion, block_id):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        pointwise_filters = int(filters * alpha)
        hidden_dim = in_channels * expansion
        self.use_residual = self.stride == 1 and in_channels == pointwise_filters

        layers = []
        # Expand phase
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
        else:
            hidden_dim = in_channels  # If expansion is 1, hidden_dim is same as in_channels

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        ])

        # Projection phase
        layers.extend([
            nn.Conv2d(hidden_dim, pointwise_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(pointwise_filters),
        ])

        self.conv = nn.Sequential(*layers)

        # Adjust input if necessary
        self.adjust = nn.Identity()
        if self.use_residual and in_channels != pointwise_filters:
            self.adjust = nn.Sequential(
                nn.Conv2d(in_channels, pointwise_filters, kernel_size=1, bias=False),
                nn.BatchNorm2d(pointwise_filters),
            )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.use_residual:
            identity = self.adjust(identity)
            out = out + identity
        return out

class MobileNetV2Like(nn.Module):
    """
    Class taken from MLSys Lab 2
    """
    def __init__(self, input_shape=(3, 32, 32), num_classes=10, alpha=1.0):
        super(MobileNetV2Like, self).__init__()
        input_channels = 32
        last_channels = 1280

        self.features = []

        # Initial convolution layer
        self.features.append(nn.Sequential(
            nn.Conv2d(input_shape[0], input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        ))

        # Build inverted residual blocks
        block_configs = [
            # filters, alpha, stride, expansion, block_id
            (16, alpha, 1, 1, 0),
            (24, alpha, 1, 6, 1),
            (24, alpha, 1, 6, 2),
            (32, alpha, 2, 6, 3),
            (32, alpha, 1, 6, 4),
            (32, alpha, 1, 6, 5),
            (64, alpha, 2, 6, 6),
            (64, alpha, 1, 6, 7),
            (64, alpha, 1, 6, 8),
            (64, alpha, 1, 6, 9),
            (96, alpha, 1, 6, 10),
            (96, alpha, 1, 6, 11),
            (96, alpha, 1, 6, 12),
            (160, alpha, 2, 6, 13),
            (160, alpha, 1, 6, 14),
            (160, alpha, 1, 6, 15),
            (320, alpha, 1, 6, 16),
        ]

        in_channels = input_channels
        for filters, alpha, stride, expansion, block_id in block_configs:
            self.features.append(InvertedResidualBlock(in_channels, filters, alpha, stride, expansion, block_id))
            in_channels = int(filters * alpha)

        # Final convolution layer
        self.features.append(nn.Sequential(
            nn.Conv2d(in_channels, last_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.ReLU(inplace=True),
        ))

        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--strategy", type=str, default="ring_all_reduce", choices=["ring_all_reduce", "param_server"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--use_wandb", type=bool, default=True)
    args = parser.parse_args()

    # Instantiate the model
    model = MobileNetV2Like(input_shape=(3, 32, 32), num_classes=10)

    transform = transforms.Compose([
        transforms.ToTensor(),  # This will also normalize from [0, 255] to [0, 1]
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    train_size = 1000
    train_indices = torch.randperm(len(train_dataset))[:train_size]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    
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