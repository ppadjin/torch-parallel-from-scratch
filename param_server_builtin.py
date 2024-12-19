# The core of this code is copied from MLSys Lab 2 and modified for this project
# Setup
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
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
import wandb
import torch.nn as nn
from data_parallel import DataParallel
from utils import get_available_gpus

def mobilenet_v2_like(input_shape=(3, 32, 32), num_classes=10):
    class InvertedResidualBlock(nn.Module):
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

    return MobileNetV2Like(input_shape=input_shape, num_classes=num_classes)

def train(model, train_loader, criterion, optimizer, device, epoch, args):
    model.train()  # Set model to training mode
    running_loss = 0.0
    total = 0
    correct = 0

    # Initialize timing variables for this epoch
    epoch_data_loading_time = 0.0
    epoch_forward_time = 0.0
    epoch_backward_time = 0.0
    epoch_optimization_time = 0.0
    epoch_other_time = 0.0

    epoch_start_time = time.time()

    # Start timing data loading
    data_loading_start = time.time()
    for inputs, labels in train_loader:
        # End timing data loading
        data_loading_end = time.time()
        epoch_data_loading_time += data_loading_end - data_loading_start

        # Start timing other operations (e.g., data transfer)
        other_start = time.time()
        # Move data and labels to the device
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.squeeze().to(device, non_blocking=True)
        # End timing other operations
        other_end = time.time()
        epoch_other_time += other_end - other_start

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        forward_start = time.time()
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)
        torch.cuda.synchronize()  # Ensure all CUDA kernels are finished
        forward_end = time.time()
        epoch_forward_time += forward_end - forward_start

        # Backward pass
        backward_start = time.time()
        loss.backward()

        optimizer.step()
        torch.cuda.synchronize()  # Ensure all CUDA kernels are finished
        backward_end = time.time()
        epoch_backward_time += backward_end - backward_start

        # Accumulate statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Start timing data loading for the next batch
        data_loading_start = time.time()


    epoch_end_time = time.time()
    epoch_total_time = epoch_end_time - epoch_start_time

    # Calculate Python overhead time
    measured_time = (
        epoch_data_loading_time
        + epoch_forward_time
        + epoch_backward_time
        + epoch_optimization_time
        + epoch_other_time
    )
    epoch_python_overhead_time = epoch_total_time - measured_time

    # Calculate average loss and accuracy
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100.0 * correct / total

    print(
        f'Epoch [{epoch + 1}/{args.epochs}] - '
        f'Loss: {epoch_loss:.4f} - '
        f'Accuracy: {epoch_acc:.2f}%'
    )

    if args.use_wandb:
        wandb.log({
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "time":time.time() - args.start_time,
            "epoch": epoch,
            "data_loading_time": epoch_data_loading_time,
            "forward_time": epoch_forward_time,
            "backward_time": epoch_backward_time,
            "all_others_time": epoch_python_overhead_time,
            "total_time": epoch_total_time
         })

    # Print timing breakdown for this epoch
    print(f"Epoch {epoch + 1} Timing Breakdown:")
    print(f"Total epoch time: {epoch_total_time:.2f} seconds")
    print(f"Data loading time: {epoch_data_loading_time:.2f} seconds ({100 * epoch_data_loading_time / epoch_total_time:.2f}%)")
    print(f"Forward pass time: {epoch_forward_time:.2f} seconds ({100 * epoch_forward_time / epoch_total_time:.2f}%)")
    print(f"Backward pass time: {epoch_backward_time:.2f} seconds ({100 * epoch_backward_time / epoch_total_time:.2f}%)")
    print(f"All others time: {epoch_python_overhead_time:.2f} seconds ({100 * epoch_python_overhead_time / epoch_total_time:.2f}%)\n")

    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move data and labels to the device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.squeeze().to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    # Calculate average test loss and accuracy
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100.0 * test_correct / test_total

    print(
        f'Test Loss: {test_loss:.4f} - '
        f'Test Accuracy: {test_acc:.2f}%\n'
    )

    return test_loss, test_acc

if __name__ == "__main__":

    # Constants
    parser = argparse.ArgumentParser(description='Parameter server built-in')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--train_size', type=int, default=1000, help='Number of samples in the training set')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of samples in the test set')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--n_gpus', type=int, default=2, help='Number of GPUs to use for training')
    parser.add_argument('--use_wandb', type=bool, default=True, help='Whether to use Weights & Biases for logging')

    args = parser.parse_args()
    # Define transforms for preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),  # This will also normalize from [0, 255] to [0, 1]
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )

    # Create subsampled indices
    train_indices = torch.randperm(len(train_dataset))[:args.train_size]
    test_indices = torch.randperm(len(test_dataset))[:args.test_size]

    # Create subsampled datasets using SubsetRandomSampler
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    # Instantiate the model
    model = mobilenet_v2_like()

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=4,  # Parallel data loading
        # pin_memory=True,  # Faster data transfer to GPU
        # prefetch_factor=2,        # Number of batches to prefetch per worker
        # persistent_workers=True   # Keep workers alive between epochs
    )

    # Move the model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr
    )

    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    available_gpus = get_available_gpus()[:args.n_gpus]
    assert len(available_gpus) == args.n_gpus, f"There are only {len(available_gpus)} available GPUs, but you requested {args.n_gpus} GPUs."
    # Check if multiple GPUs are available
    if len(available_gpus) > 1:
        print("Using", available_gpus, "GPUs for training.")
        model = nn.DataParallel(model, device_ids=available_gpus)

    # Move model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("Param server builtin is running")

    if args.use_wandb:
        wandb.init(project="param_server_builtin")
        wandb.config.update(args)

    # Main loop: perform training and evaluation

    args.start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer, device, epoch, args
        )