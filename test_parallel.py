from multiprocess import Pool, cpu_count
import time
from tqdm import tqdm
import numpy as np

def simple_work(x):
    # Simulate some numpy work
    arr = np.random.rand(100, 100)
    return np.sum(arr @ arr)

n = 100000
print(f'Testing with {cpu_count()} cores...')

# Sequential
start = time.time()
results = list(tqdm(
    (simple_work(i) for i in range(n)), 
    total=n, 
    desc="Processing sequentially"
))
seq_time = time.time() - start
print(f'Sequential: {n/seq_time:.1f} it/s')

# Parallel
start = time.time()
with Pool(cpu_count()) as pool:
    results = list(tqdm(
        pool.imap(simple_work, range(n)), 
        total=n, 
        desc="Processing in parallel"
    ))
par_time = time.time() - start
print(f'Parallel: {n/par_time:.1f} it/s')
print(f'Speedup: {seq_time/par_time:.1f}x')

# Parallel little cpus
start = time.time()
with Pool(5) as pool:
    results = list(tqdm(
        pool.imap(simple_work, range(n)), 
        total=n, 
        desc="Processing in parallel but reduced"
    ))
par_time = time.time() - start
print(f'Parallel but reduced: {n/par_time:.1f} it/s')
print(f'Speedup but reduced: {seq_time/par_time:.1f}x')