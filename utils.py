
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