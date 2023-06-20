import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, writers

from tqdm import tqdm
import json
from datetime import datetime

from utils import *
from algo import PEDS_SGD


rast1 = lambda x: rastrigin_function(x, 3)


def run_optimize(N, m, objective, optimizer, early_stop_norm=1e-4, verbose=False):
    x_traj = []
    
    x = generate_random(-4, 4, (N,m))
    if verbose: print("Initial condition:", x)
    x.requires_grad = True

    optim = optimizer([x])

    # Optimize the objective function
    for _ in range(int(1e4)):
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

optimizer = partial(torch.optim.SGD, lr=1e-4)

# N = m = 1
result = run_optimize(1, 2, rast1, optimizer)
# result = experiment(1, rast1, optimizer, int(1e4), 0, 0.1)

