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
from utils import get_available_gpus, get_data
from model import MobileNetV2Like

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

    # Load CIFAR-10 dataset
    train_subset = get_data(train_size=args.train_size)
    model = MobileNetV2Like(input_shape=(3, 32, 32), num_classes=10)

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True
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