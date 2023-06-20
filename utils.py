import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from functools import partial
from matplotlib.animation import FuncAnimation, writers
from joblib import Parallel, delayed
import multiprocessing
import random
import time


from tqdm import tqdm

# Input should be a torch tensor
def rastrigin_function(x, A):
    # broadcast over the last dimension
    return torch.sum(A + x ** 2 - A * torch.cos(2 * torch.pi * x), dim=-1)


def generate_random(upper, lower, shape):
    return lower + torch.rand(shape) * (upper - lower)

def run_optimize(N, objective, optimizer, early_stop_norm=1e-4, verbose=False):
    x_traj = []
    
    x = generate_random(-4, 4, N)
    if verbose: print("Initial condition:", x)
    x.requires_grad = True

    optim = optimizer([x])

    # Optimize the objective function
    for _ in range(int(1e4)):
        # Wait, then I can just sum all in objective, because why not
        y = torch.sum(objective(x)) # summation doesn't affect the gradient
        optim.zero_grad()
        y.backward()
        
        # If the step is small, early stop
        if x_traj and np.linalg.norm(x.detach().numpy() - x_traj[-1]) < early_stop_norm:
            if verbose: print("Stopping gradient:", x.grad)
            break
            
        x_traj.append(x.detach().numpy().copy())
        optim.step()
    
    return x_traj

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def experiment(
    N,
    objective,
    optimizer,
    sample_size,
    optimum,
    tol,
    seed_value=42,
):
    start_time = time.time()

    set_seed(seed_value)
    
    x_last = []
    losses = []

    # The number of threads is equalt to the number of cores
    num_cores = multiprocessing.cpu_count()

    x_trajs = Parallel(n_jobs=num_cores)(delayed(run_optimize)(N, objective, optimizer) for _ in range(sample_size))
    
    x_last = [traj[-1] for traj in x_trajs]
    losses = [objective(torch.tensor(x)) for x in x_last]

    succ = [np.linalg.norm(x_end - optimum) < tol for x_end in x_last]

    return {
        'x_trajs': x_trajs,
        'x_last': x_last,
        'losses': losses,
        'success_rate': sum(succ)/sample_size,
        'seed_value': seed_value,
        'total_time': time.time() - start_time
    }

