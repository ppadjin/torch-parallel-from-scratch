import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Replace with your available GPU indices

import subprocess

def get_available_gpus(th=10):
    """
    Avoid using someone else's GPU by checking for available GPUs. Criteria is that I only use GPUs with less than th% utilization.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    
    )
    assert result.returncode == 0 # check validit

    available_gpus = []
    for line in result.stdout.strip().split("\n"):
        gpu_index, gpu_utilization = line.split(", ")
        gpu_index = int(gpu_index)
        gpu_utilization = int(gpu_utilization)
        
        if gpu_utilization < th: available_gpus.append(gpu_index)

    return available_gpus

def gpu_worker(gpu_id, model, data, results_queue):
    try:
        # Set the device for this process
        torch.cuda.set_device(gpu_id)
        
        # Move model and data to this GPU
        model = model.to(f'cuda:{gpu_id}')
        data = data.to(f'cuda:{gpu_id}')
        
        # Perform computation
        output = model(data)
        
        # Store the result in the queue
        results_queue.put((gpu_id, output.detach().cpu()))
    except Exception as e:
        print(f"Error in GPU {gpu_id}: {e}")

def train(model, gpu_id, data_loader, optimizer, loss_fn):
    torch.cuda.set_device(gpu_id)
    # Construct data_loader, optimizer, etc.
    for data, labels in data_loader:
        optimizer.zero_grad()
        loss_fn(model(data), labels).backward()
        optimizer.step()  # This will update the shared parameters

        print("Training on GPU", torch.cuda.current_device())


if __name__ == "__main__":
    # Define the model and data
    model = nn.Linear(10, 10)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.share_memory()
    data = torch.randn(128, 10)  # Batch of 32 samples
    labels = torch.randint(0, 10, (128,))  # 10 classes
    mp.set_start_method("spawn")

    available_gpus = get_available_gpus()
    # use maximum 4 GPUs
    devices = available_gpus[:4]
    #devices = [1] # debugging with 1
    num_gpus = len(devices)
    print(f"Using {num_gpus} GPUs: {devices}")

    
    # Split data into chunks for each GPU
    data_chunks = torch.chunk(data, len(devices))
    label_chunks = torch.chunk(labels, len(devices))

    # Create a DataLoader for each GPU
    data_loaders = []
    for i, (data_chunk, label_chunk) in enumerate(zip(data_chunks, label_chunks)):
        data_loaders.append(DataLoader(
            torch.utils.data.TensorDataset(data_chunk, label_chunk),
            batch_size=32,
            shuffle=True
        ))

    
    # Create a multiprocessing queue to collect results
    results_queue = mp.Queue()
    
    # Launch a process for each GPU
    processes = []
    for i, (gpu_id, data_chunk) in enumerate(zip(devices, data_chunks)):
        #p = mp.Process(target=gpu_worker, args=(gpu_id, model, data_chunk, results_queue))
        p = mp.Process(target=train, args=(model, gpu_id, data_loaders[i], optimizer, loss_fn))
        processes.append(p)
        p.start()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    pass