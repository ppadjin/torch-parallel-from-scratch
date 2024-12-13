import os
import torch
import subprocess
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.multiprocessing import Manager
from torch.utils.data import DataLoader, TensorDataset
from torch.multiprocessing import Lock
from example import get_available_gpus

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Replace with your available GPU indices
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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



def parameter_server(model_queue, gradients_queue, workers_done, num_workers, epochs, param_server_device, data_loader, lock):
    """
    Parameter server to aggregate gradients and update the model.
    """

    n_class = 4
    lr = 0.001
    torch.cuda.set_device(param_server_device)
    shared_model = nn.Sequential(nn.Linear(10, 30), nn.ReLU(), nn.Linear(30, n_class)).to(param_server_device)  # A local copy of the model
    shared_model = shared_model.to(f"cuda:{param_server_device}")
    optimizer = optim.SGD(shared_model.parameters(), lr=lr)

    accs = []
    for epoch in range(epochs):
        print(f"Server: Starting epoch {epoch + 1}")
        for _ in range(num_workers):
            model_queue.put(shared_model.state_dict())

        # Collect gradients from all workers
        for _ in range(num_workers):
            try:
                worker_grads, rank = gradients_queue.get()
            except Exception as e:
                print(f"Server: Error getting gradients from worker: {e}")
                continue
            with torch.no_grad():
                print(f"Server: Received gradients from worker {rank}")
            
                for param, grad in zip(shared_model.parameters(), worker_grads):
                    if grad is not None:
                        grad = grad.to(param.device)
                        param.grad = grad if param.grad is None else param.grad + grad

        # Average gradients
    
        with torch.no_grad():
            for param in shared_model.parameters():
                if param.grad is not None:
                    param.grad /= num_workers
            
        # calculate current training accuracy
        acc = 0
        for batch, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(f"cuda:{param_server_device}"), targets.to(f"cuda:{param_server_device}").long()
            outputs = shared_model(inputs)
            _, predicted = torch.max(outputs, 1)
            acc += (predicted == targets).sum().item()

        acc /= len(data_loader.dataset)
        accs.append(acc)
        # Update model
        optimizer.step()
        optimizer.zero_grad()

        # Notify workers that the model is updated
        workers_done.value += 1
    
    print(f"Server: Training completed. Accuracies: {accs}")

def worker(rank, model_queue, gradients_queue, workers_done, data_loader, loss_fn, epochs, lock):
    """
    Worker process to perform training and send gradients to the server.
    """
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    print(f"Worker {device}: Starting")
    n_class = 4 
    local_model = nn.Sequential(nn.Linear(10, 30), nn.ReLU(), nn.Linear(30, n_class)).to(device)  # A local copy of the model

    
    for epoch in range(epochs):

        while workers_done.value < epoch:
            pass
        # Load updated model from shared model
        
        curr_state_dict = model_queue.get()
        local_model.load_state_dict(curr_state_dict)        

        local_model.train()
        local_model.zero_grad(set_to_none=True)
        for batch, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device).long()
            # Forward and backward pass
            outputs = local_model(inputs)
            outputs = F.log_softmax(outputs, dim=-1)
            loss = loss_fn(outputs, targets)
            loss.backward()

        # Collect gradients
        gradients = [param.grad.clone() for param in local_model.parameters()]
        gradients_queue.put((gradients, rank))

        
        print(f"Worker {device}: Completed epoch {epoch + 1}")
    
    # Wait for the server to finish updating the model
    while workers_done.value < epochs:
        pass


def train_single_process(model, data_loader, loss_fn, optimizer, epochs):
    # train the model in a single process, for comparison
    accs = []
    for epoch in range(epochs):
        for batch, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.long()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch + 1} completed")
        acc = 0
        i = 0
        for batch, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            acc += (predicted == targets).sum().item()
            i += len(targets)
        acc /= len(data_loader.dataset)
        accs.append(acc)
    print(f"Training completed. Accuracies: {accs}")


def main():
    mp.set_start_method('spawn')

    # Parameters
    num_workers = 2
    epochs = 2000
    lr = 0.001
    n_class = 4
    N = 100
    sequential = False

    inputs = torch.randn(N, 10)
    # split targets into 4 classes based on the input magnitude (if |input| < 0.1, class = 0, etc.)
    targets = torch.zeros(N)
    for i in range(N):
        if torch.norm(inputs[i]) < 2.0:
            targets[i] = 0
        elif torch.norm(inputs[i]) < 3.0:
            targets[i] = 1
        elif torch.norm(inputs[i]) < 4.0:
            targets[i] = 2
        else:
            targets[i] = 3

    print(f'Value counts of targets: {targets.unique(return_counts=True)}')

    # Dummy dataset
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=10)

    # Model and loss function
    loss_fn = nn.CrossEntropyLoss()

    # Shared objects for communication
    gradients_queue = mp.Queue()
    workers_done = mp.Value('i', 0)

    #shared_model = nn.Sequential(nn.Linear(10, 30), nn.ReLU(), nn.Linear(30, n_class))
    #shared_model.share_memory()

    #optimizer = optim.SGD(shared_model.parameters(), lr=lr)
    model_queue = mp.Queue()

    if not sequential:
        # Spawn processes
        processes = []
        gpu_ids = get_available_gpus()
        gpu_ids = gpu_ids[:num_workers + 1] # 1 for server + rest for workers
        lock = Lock()

        processes.append(mp.Process(target=parameter_server, args=(model_queue, gradients_queue, workers_done, num_workers, epochs, gpu_ids[0], data_loader, lock)))
        for rank in gpu_ids[1:]:
            processes.append(mp.Process(target=worker, args=(rank, model_queue, gradients_queue, workers_done, data_loader, loss_fn, epochs, lock)))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    else:
        # Train in a single process for comparison
        print('-'*50)
        print("Training in a single process")
        model = nn.Sequential(nn.Linear(10, 30), nn.ReLU(), nn.Linear(30, n_class))
        optimizer = optim.SGD(model.parameters(), lr=lr)


        train_single_process(model, data_loader, loss_fn, optimizer, epochs)

if __name__ == "__main__":
    main()
