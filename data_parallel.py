import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import comm
from abc import ABC, abstractmethod
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Replace with your available GPU indices


class DataParallel(ABC):
    def __init__(self, gpu_ids):
        self.gpu_ids = gpu_ids

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
    
    def backprop_epoch(self, model, dataloader, device):
        """
        Perform one backprop epoch and return gradients
        """
        model.train()
        model.zero_grad(set_to_none=True)
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)
            outputs = F.log_softmax(outputs, dim=-1)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
        return [param.grad.clone() for param in model.parameters()] # collecting gradients


class ParameterServer(DataParallel):
    """
    This class implements a vanilla parameter server strategy for data parallelism.
    One GPU is used as a parameter server, while the rest are used as workers.
    """

    def __init__(self, dataloader, num_workers=3):
        gpu_ids = DataParallel.get_available_gpus()
        gpu_ids = gpu_ids[:num_workers + 1]
        self.num_workers = num_workers
        super().__init__(gpu_ids)
        self.server_device = gpu_ids[0]
        self.workers = gpu_ids[1:]
        self.dataloader = dataloader

    def train(self, model, lr = 0.001, epochs = 1000, loss_fn = nn.CrossEntropyLoss()):

        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn
        mp.set_start_method("spawn")

        self.model_queue = mp.Queue()
        self.gradients_queue = mp.Queue()
        self.workers_done = mp.Value("i", 0)

        processes = []
        for gpu_id in self.gpu_ids:
            p = mp.Process(target=self.run, args=(gpu_id, model))
            processes.append(p)
            p.start()
    
        # Wait for all processes to finish
        for p in processes:
            p.join()
    
    def run(self, gpu_id, model):
        
        if gpu_id == self.server_device:
            # Perform parameter update
            self.run_server(gpu_id, model)
        else:
            # Perform computation
            self.run_worker(gpu_id, model)
        
    def run_server(self, gpu_id, shared_model):
        """
        A method to run the parameter server. It gives the new model to
        other devices and collects gradients from them, averaging them.
        """
        assert gpu_id == self.server_device

        torch.cuda.set_device(self.server_device)
        shared_model = shared_model.to(f"cuda:{self.server_device}")
        optimizer = optim.SGD(shared_model.parameters(), lr=self.lr)

        accs = []
        for epoch in range(self.epochs):
            print(f"Server: Starting epoch {epoch + 1}")
            for _ in range(self.num_workers):
                self.model_queue.put(shared_model.state_dict())

            # Collect gradients from all workers
            for _ in range(self.num_workers):
                try:
                    worker_grads, rank = self.gradients_queue.get()
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
                        param.grad /= self.num_workers
                
            # calculate current training accuracy
            acc = self.calculate_acc(shared_model)
            accs.append(acc)
            # Update model
            optimizer.step()
            optimizer.zero_grad()

            # Notify workers that the model is updated
            self.workers_done.value += 1
        
        print(f"Server: Training completed. Accuracies: {accs}")
        
    def run_worker(self, gpu_id, shared_model):
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

        print(f"Worker {device}: Starting")
        local_model = shared_model.to(device)
        for epoch in range(self.epochs):

            while self.workers_done.value < epoch:
                pass
            # Load updated model from shared model
            
            curr_state_dict = self.model_queue.get()
            local_model.load_state_dict(curr_state_dict)        

            gradients = self.backprop_epoch(local_model, self.dataloader, device)
            self.gradients_queue.put((gradients, gpu_id))

            print(f"Worker {device}: Completed epoch {epoch + 1}")
        
        # Wait for the server to finish updating the model
        while self.workers_done.value < self.epochs:
            pass        
    
    def calculate_acc(self, model):
        acc = 0
        for _, (inputs, targets) in enumerate(self.dataloader):
            inputs, targets = inputs.to(f"cuda:{self.server_device}"), targets.to(f"cuda:{self.server_device}").long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            acc += (predicted == targets).sum().item()

        acc /= len(self.dataloader.dataset)
        return acc


class RingAllReduce(DataParallel):
    def __init__(self, dataloader, num_gpus=3):
        gpu_ids = DataParallel.get_available_gpus()
        gpu_ids = gpu_ids[:num_gpus]
        self.num_workers = num_gpus
        super().__init__(gpu_ids)
        
        self.dataloader = dataloader

    def train(self, model, lr = 0.001, epochs = 1000, loss_fn = nn.CrossEntropyLoss()):
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn
        mp.set_start_method("spawn")
        self.barrier = mp.Barrier(self.num_workers)
        processes = self.create_ring_processes(model, self.num_workers)

        for p in processes:
            p.join()

    def train_ring_node(self, process_id, input_queue, output_queue):
        """
        Function to simulate a process in a ring communication pattern.
        
        process_id: Unique identifier for this process
        num_processes: Total number of processes
        input_queue: Queue to receive data from the previous process
        output_queue: Queue to send data to the next process
        """

        # first train for an epoch and get grads
        gradients = self.backprop_epoch(model, self.dataloader, self.gpu_ids[process_id])
        
        # share-reduce
        for i in range(self.num_workers-1):
            curr_send_idx = (process_id + i) % self.num_workers
            rec_idx = (process_id - i - 1) % self.num_workers
            curr_chunk = np.array_split(gradients, self.num_workers)[curr_send_idx]
            output_queue.put(curr_chunk)
            received_chunk = input_queue.get()
            
            #gradients += received_chunk
            self.barrier.wait()


        '''for i in range(10):
            try:
                # Receive data from the previous process
                received_data = input_queue.get()
                print(f"Process {process_id} received: {received_data}")
                
                # Modify the data or create a new message
                new_data = f"{received_data} -> Process {process_id}"
                
                # Send to the next process
                output_queue.put(new_data)
                print(f"Process {process_id} sent: {new_data}")
                
                # Optional: break condition (e.g., after a certain number of iterations)
                if process_id == 0 and "Process 0 -> full circle" in new_data:
                    break
            
            except:
                print(f"Process {process_id} timed out waiting for input")
                break'''
        
        print(f"Process {process_id} exiting...")

    def create_ring_processes(self, model, num_processes):
        """
        Create N processes in a ring communication pattern.
        
        model: Model to train
        num_processes: Number of processes to create
        
        Returns:
        List of process objects
        """
        queues = [mp.Queue() for _ in range(num_processes)]
        
        # Create and start processes
        processes = []
        for i in range(num_processes):
            
            input_queue = queues[(i-1+num_processes)%num_processes] # formula for next node in ring
            output_queue = queues[i]
            
            p = mp.Process(
                target=self.train_ring_node, 
                args=(i, input_queue, output_queue))
            
            p.start()
            processes.append(p)
        
        return processes



if __name__ == "__main__":
    N = 100
    n_class = 4
    inputs = torch.randn(N, 10)
    targets = torch.randint(0, n_class, (N,))
    dataset = torch.utils.data.TensorDataset(inputs, targets)

    model = nn.Sequential(nn.Linear(10, 30), nn.ReLU(), nn.Linear(30, n_class))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    '''param_server = ParameterServer(dataloader, num_workers=1)
    param_server.train(model, epochs=2000)'''


    ring_all_reduce = RingAllReduce(dataloader, num_gpus=3)
    ring_all_reduce.train(model, epochs=2000)
    