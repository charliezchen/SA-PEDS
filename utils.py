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

from algo import MultipleCopy, Rastrigin

from tqdm import tqdm

# Input should be a torch tensor
def rastrigin_function(x, A):
    # broadcast over the last dimension
    return torch.sum(A + x ** 2 - A * torch.cos(2 * torch.pi * x), dim=-1)


def generate_random(upper, lower, shape):
    return lower + torch.rand(shape) * (upper - lower)


def run_optimize(model_class, optimizer_class, minimal_step=1e-4, verbose=False):
    x_traj = []
    y_traj = []

    model = model_class()
    optimizer = optimizer_class(model.parameters())


    # Optimize the objective function
    for _ in range(int(1e6)):
        # with torch.profiler.profile(record_shapes=True) as prof:
            # This is a numpy matrix of shape N x m
            x = model.x.detach().numpy().copy()
            if x_traj:
                steps = [np.linalg.norm(x[i]-x_traj[-1][i]) for i in range(len(x))]
                if np.max(steps) < minimal_step:
                    if verbose: print("The maximum step is:", np.max(steps), "\nQuit early")
                    break
            x_traj.append(x)


            # This is a torch tensor of length N
            y = model()
            y_traj.append([yi.detach().numpy().copy() for yi in y])

            loss = torch.sum(y)
            optimizer.zero_grad()
            loss.backward()
            
            model.peds_step()
            optimizer.step()
        # print(prof.key_averages().table(sort_by="cpu_time_total"))
        # exit(0)

    return x_traj, y_traj

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
    model_class,
    optimizer_class,
    sample_size,
    optimum,
    tol=0.1,
    seed_value=42,
):
    start_time = time.time()

    set_seed(seed_value)

    # The number of threads is equalt to the number of cores
    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores) \
        (delayed(run_optimize)(model_class, optimizer_class) for _ in range(sample_size))
    
    # results = [(run_optimize)(model_class, optimizer_class)]

    list_x_traj = [res[0] for res in results]
    list_y_traj = [res[1] for res in results]

    losses = [y_traj[-1] for y_traj in list_y_traj]
    mean_loss = np.mean(losses)

    last_x = [x_traj[-1] for x_traj in list_x_traj]
    num_succ = [np.min(np.linalg.norm(x_end - optimum, axis=1)) < tol for x_end in last_x]

    return {
        # 'list_x_traj': list_x_traj,
        # 'list_y_traj': list_y_traj,
        'last_x': last_x,
        # 'losses': losses,
        'mean_loss': mean_loss,
        # 'num_succ': num_succ,
        'success_rate': sum(num_succ)/sample_size,
        # 'seed_value': seed_value,
        'total_time': time.time() - start_time
    }

