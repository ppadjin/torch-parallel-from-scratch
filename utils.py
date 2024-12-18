import subprocess

def divide_to_chunks(l, n, return_slices=False):
    """
    Properly divides list l into n chunks.
    """
    chunk_size = len(l) // n
    rem = len(l) % n
    
    chunks = []
    chunk_slices = []
    i = 0
    for _ in range(n):
        if rem > 0:
            chunk = l[i:i+chunk_size+1]
            chunk_slices.append(slice(i, i+chunk_size+1))
            rem -= 1
            i += chunk_size + 1
        else:
            chunk = l[i:i+chunk_size]
            chunk_slices.append(slice(i, i+chunk_size))
            i += chunk_size
        chunks.append(chunk)
    
    return chunks if not return_slices else (chunks, chunk_slices)


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