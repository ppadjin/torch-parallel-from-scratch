import os
import subprocess
import time
import numpy as np
import torch
import wandb
import torch.nn as nn
import torch.multiprocessing as mp
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from utils import divide_to_chunks
from datamanager import DataManager

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Replace with your available GPU indices


class DataParallel(ABC):
    def __init__(self, gpu_ids, use_wandb=True, wandb_setup={}):
        self.gpu_ids = gpu_ids
        self.use_wandb = use_wandb
        self.wandb_config = wandb_setup['config']
        self.wandb_project = wandb_setup['project']

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
    
    def backprop_epoch(self, model, dataloader, process_id):
        """
        Perform one backprop epoch and return gradients
        """
        device = f"cuda:{process_id}"
        model.train()
        model.zero_grad(set_to_none=True)
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)
            outputs = F.log_softmax(outputs, dim=-1)
            loss = self.loss_fn(outputs, targets)
            if self.use_wandb and process_id == 0:
                wandb.log({"training loss": loss.item()})
            loss.backward()
        return [param.grad.clone() for param in model.parameters()] # collecting gradients

    def calculate_acc(self, model, dataloader, process_id):
        acc = 0
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(f"cuda:{process_id}"), targets.to(f"cuda:{process_id}").long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            acc += (predicted == targets).sum().item()

        acc /= len(dataloader.dataset)
        return acc

class ParameterServer(DataParallel):
    """
    This class implements a vanilla parameter server strategy for data parallelism.
    One GPU is used as a parameter server, while the rest are used as workers.
    """

    def __init__(self, datamanager, num_gpus=3, *args, **kwargs):
        self.num_workers = num_gpus - 1
        gpu_ids = DataParallel.get_available_gpus()
        gpu_ids = gpu_ids[:num_gpus]
        super().__init__(gpu_ids, *args, **kwargs)
        self.server_device = gpu_ids[0]
        self.workers = gpu_ids[1:]
        self.gpu_process_map = {gpu_id: i for i, gpu_id in enumerate(gpu_ids)} # easy way to change between gpu id and process id
        self.datamanager = datamanager
        self.init_wandb = False

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

        self.start_time = time.time()

        torch.cuda.set_device(self.server_device)
        shared_model = shared_model.to(f"cuda:{self.server_device}")
        optimizer = optim.SGD(shared_model.parameters(), lr=self.lr)

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
            acc = self.calculate_acc(shared_model, self.datamanager.get_dataloader(gpu_id), gpu_id)
            tr_loss = self.calculate_training_loss(shared_model, self.datamanager.get_dataloader(gpu_id), gpu_id)
            if self.use_wandb:
                if not self.init_wandb:
                    wandb.init(project="test", group='DDP')
                    self.init_wandb = True
                wandb.log({"training acc": acc})
                wandb.log({"training loss": tr_loss})
                wandb.log({"Time": time.time() - self.start_time})
                print(f"Server: Epoch {epoch + 1}, acc: {acc}")
                print(f"Server: Epoch {epoch + 1}, loss: {tr_loss}")
            # Update model
            optimizer.step()
            optimizer.zero_grad()

            # Notify workers that the model is updated
            self.workers_done.value += 1
                
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
            gradients = self.backprop_epoch(local_model, self.datamanager.get_dataloader(self.gpu_process_map[gpu_id]), gpu_id)
            
            self.gradients_queue.put((gradients, gpu_id))

            print(f"Worker {device}: Completed epoch {epoch + 1}")
        
        # Wait for the server to finish updating the model
        while self.workers_done.value < self.epochs:
            pass        

    def calculate_training_loss(self, model, dataloader, process_id):
        """
        Calculate the training loss for a given model and dataloader.
        """
        loss = 0
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(f"cuda:{process_id}"), targets.to(f"cuda:{process_id}").long()
            outputs = model(inputs)
            outputs = F.log_softmax(outputs, dim=-1)
            loss += self.loss_fn(outputs, targets).item()
        return loss


class RingAllReduce(DataParallel):
    """
    This class implements a ring all-reduce strategy for data parallelism.

    Example:
        A B C D

        i=0:

        A:send a0, rec d3
        B:send b1, rec a0
        C:send c2, rec b1
        D:send d3, rec c2

        i=1:

        A: send a3+d3, rec c2+d2
        B: send a0+b0, rec a3+d3
        C: send b1+c1, rec a0+b0
        D: send c2+d2, rec b1+c1

        i = 2:

        A: send a2+c2+d2, rec b1+c1+d1
        B: send a3+b3+d3, rec a2+c2+d2
        C: send a0+b0+c0, rec a3+b3+d3
        D: send b1+c1+d1, rec a0+b0+c0

        ___

        Share only:

        A: a1+b1+c1+d1
        B: a2+b2+c2+d2
        C: a3+b3+c3+d3
        D: a0+b0+c0+d0
    """
    def __init__(self, datamanager: DataManager, num_gpus=3, *args, **kwargs):
        gpu_ids = DataParallel.get_available_gpus()
        gpu_ids = gpu_ids[:num_gpus]
        self.num_workers = num_gpus
        super().__init__(gpu_ids, *args, **kwargs)
        
        self.datamanager = datamanager

        wandb.init(project=self.wandb_project, group='DDP', config=self.wandb_config)

    def train(self, model, lr = 0.1, epochs = 1000, loss_fn = nn.CrossEntropyLoss()):
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.barrier = mp.Barrier(self.num_workers)
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr)
        processes = self.create_ring_processes(model, self.num_workers)

        for p in processes:
            p.join()

    def train_ring_node(self, process_id, model, input_queue, output_queue):
        """
        Function to simulate a process in a ring communication pattern.
        
        process_id: Unique identifier for this process
        num_processes: Total number of processes
        input_queue: Queue to receive data from the previous process
        output_queue: Queue to send data to the next process
        """
        model.train()
        torch.cuda.set_device(self.gpu_ids[process_id])
        model = model.to(f"cuda:{self.gpu_ids[process_id]}")
        for epoch in range(self.epochs):
            # first train for an epoch and get grads
            gradients = self.backprop_epoch(model, self.datamanager.get_dataloader(process_id), process_id)
            
            # share-reduce
            for i in range(self.num_workers-1):
                curr_send_idx = (process_id - i) % self.num_workers
                rec_idx = (process_id - i - 1) % self.num_workers
                chunks, slices = divide_to_chunks(gradients, self.num_workers, return_slices=True)
                output_queue.put(chunks[curr_send_idx])
                received_chunk = input_queue.get()
                for idx in range(slices[rec_idx].start, slices[rec_idx].stop):
                    gradients[idx] += received_chunk[idx - slices[rec_idx].start].to(f"cuda:{self.gpu_ids[process_id]}")

            self.barrier.wait()

            # share-only stage to get the final grads
            for i in range(self.num_workers-1):
                # now we need to shift them by one, because A contains the correct 1-chunks, B contains the correct 2-chunks, etc.

                curr_send_idx = (process_id - i + 1) % self.num_workers
                rec_idx = (process_id - i) % self.num_workers
                chunks, slices = divide_to_chunks(gradients, self.num_workers, return_slices=True)
                output_queue.put(chunks[curr_send_idx])
                received_chunk = input_queue.get()
                for idx in range(slices[rec_idx].start, slices[rec_idx].stop):
                    gradients[idx] = received_chunk[idx - slices[rec_idx].start].to(f"cuda:{self.gpu_ids[process_id]}")
                        
            # update grads
            for param, grad in zip(model.parameters(), gradients):
                if grad is not None:
                    grad = grad.to(param.device)
                    param.grad = grad if param.grad is None else param.grad + grad

            self.barrier.wait()

            # update model
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            acc = self.calculate_acc(model, self.datamanager.get_dataloader(process_id), process_id)
            print(f"Process {process_id}: Epoch {epoch + 1}, acc: {acc}")
            if self.use_wandb and process_id == 0:
                wandb.log({"training acc": acc})

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
                args=(i, model, input_queue, output_queue))
            
            p.start()
            processes.append(p)
        
        return processes

    def test_gradient_validity(self, process_id):
        """
        This is a helper function to test if the gradients are being shared correctly.
        """
        process_val = 10**(-process_id)
        return [
            (torch.ones(30,10)*process_val).to(f"cuda:{self.gpu_ids[process_id]}"),
            (torch.ones(30)*process_val).to(f"cuda:{self.gpu_ids[process_id]}"),
            (torch.ones(4,30)*process_val).to(f"cuda:{self.gpu_ids[process_id]}"),
            (torch.ones(4)*process_val).to(f"cuda:{self.gpu_ids[process_id]}"),
        ]

if __name__ == "__main__":
    N = 3000
    n_class = 4
    inputs = torch.randn(N, 10)
    targets = torch.randint(0, n_class, (N,))
    num_gpus = 4
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    strategy = "param_server"
    #strategy = "ring_all_reduce"

    # wandb
    wandb.login()
    wandb.init(project="test", 
                config={
                    "n_gpus": num_gpus,
                    "batch_size": 32,
                    "epochs": 1000,
                    "strategy": strategy}, 
                group='DDP')


    

    model = nn.Sequential(nn.Linear(10, 30), nn.ReLU(), nn.Linear(30, n_class))

    #ring_all_reduce = RingAllReduce(datamanager, num_gpus=num_gpus)
    #ring_all_reduce.train(model, epochs=2000)
    
    if strategy == "ring_all_reduce":
        datamanager = DataManager(dataset, num_gpus, strategy=strategy)
        ring_all_reduce = RingAllReduce(datamanager, num_gpus=num_gpus, use_wandb=True)
        ring_all_reduce.train(model, epochs=100)
    elif strategy == "param_server":
        datamanager = DataManager(dataset, num_gpus, strategy=strategy)
        param_server = ParameterServer(datamanager, num_gpus=num_gpus, use_wandb=True)
        param_server.train(model, epochs=100)