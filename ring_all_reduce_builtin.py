import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_data
import wandb
from model import MobileNetV2Like

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)

    # Set device for this process
    torch.cuda.set_device(rank)
    if args.use_wandb and rank == 0:
        wandb.init(project=args.wandb_project_name, config=args.wandb_config)
    start_time = time.time()

    train_dataset = get_data(args.train_size)

    #sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    # Define the model, loss function, and optimizer
    model = MobileNetV2Like(input_shape=(3, 32, 32), num_classes=10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        ddp_model.train()
        #sampler.set_epoch(epoch)  # Ensure proper shuffling in each epoch
        epoch_loss = 0.0
    
        data_loading_start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)
            data_loading_time = time.time() - data_loading_start_time

            epoch_calc_start_time = time.time()
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()

            epoch_calc_time = time.time() - epoch_calc_start_time

            optim_start_time = time.time()
            optimizer.step()
            optim_time = time.time() - optim_start_time

            epoch_loss += loss.item()

        epoch_time = time.time() - start_time
        
        # calculate accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in train_loader:
                images, labels = data
                images, labels = images.to(rank), labels.to(rank)
                outputs = ddp_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Rank {rank}, Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}, Accuracy: {100 * correct / total}")

        acc = 100 * correct / total
        if args.use_wandb and rank == 0:

            wandb.log({"train_loss": epoch_loss / len(train_loader), "train_accuracy": acc, "data_loading_time": data_loading_time, "calculation_time": epoch_calc_time, "optim_time": optim_time, "epoch_time": epoch_time})



    cleanup()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--train_size', type=int, default=1000, help='Number of samples in the training set')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for training')
    parser.add_argument('--n_gpus', type=int, default=4, help='Number of GPUs to use for training')
    parser.add_argument('--use_wandb', type=bool, default=True, help='Whether to use Weights & Biases for logging')
    args = parser.parse_args()


    available_num_gpus = torch.cuda.device_count()
    world_size = min(args.n_gpus, available_num_gpus)

    if args.use_wandb:
        wandb.login()
        wandb_config = {
            "n_gpus": world_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_size": args.train_size,
        }
        wandb_project_name = 'mobilenet'
        args.wandb_project_name = wandb_project_name
        args.wandb_config = wandb_config

    torch.multiprocessing.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
