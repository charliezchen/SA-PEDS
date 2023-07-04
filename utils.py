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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Optimize using PEDS method')


    # Add arguments to the parser
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--independent', action='store_true')
    
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--alpha-inc', type=float, default=0)
    
    parser.add_argument('-m', type=int)
    parser.add_argument('--N', type=int)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sample-size', '-sz', type=float, default=1e4)


    parser.add_argument('--test-function', type=str, default='Rastrigin')
    # parser.add_argument('-A', type=int, default=3) # TODO: Get rid of this
    parser.add_argument('--shift', type=float, default=0)

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--debug', action='store_true')

    # Saving
    parser.add_argument('--folder', type=str, required=True)

    args = parser.parse_args()

    args.sample_size = int(args.sample_size)
    return args

# Input should be a torch tensor
def rastrigin_function(x, A):
    # broadcast over the last dimension
    return torch.sum(A + x ** 2 - A * torch.cos(2 * torch.pi * x), dim=-1)


def generate_random(upper, lower, shape):
    return lower + torch.rand(shape) * (upper - lower)


def run_optimize(model_class, optimizer_class, minimal_step=1e-4, save_traj = False, verbose=False):
    x_traj = []
    y_traj = []
    last_x, last_y = None, None

    model = model_class()
    optimizer = optimizer_class(model.parameters())


    # Optimize the objective function
    for _ in range(int(1e5)): #TODO: don't hard code max_iter
        # with torch.profiler.profile(record_shapes=True) as prof:
            # This is a numpy matrix of shape N x m
            x = model.x.detach().numpy().copy()
            if last_x is not None:
                steps = [np.linalg.norm(x[i]-last_x[i]) for i in range(len(x))]
                if np.max(steps) < minimal_step:
                    if verbose: print("The maximum step is:", np.max(steps), "\nQuit early")
                    break
            last_x = x


            # This is a torch tensor of length N
            y = model()
            last_y = y.detach().numpy().copy()
            # y_traj.append([yi.detach().numpy().copy() for yi in y])

            loss = torch.sum(y)
            optimizer.zero_grad()
            loss.backward()
            
            model.peds_step()
            optimizer.step()

            if save_traj:
                x_traj.append(last_x)
                y_traj.append(model.x.grad.detach().numpy().copy())

        # print(prof.key_averages().table(sort_by="cpu_time_total"))
        # exit(0)

    return last_x, last_y, x_traj, y_traj

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
    save_traj=False,
    seed_value=42,
    debug=False
):
    start_time = time.time()

    set_seed(seed_value)

    # The number of threads is equalt to the number of cores
    if not debug:
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = 1

    
    results = Parallel(n_jobs=num_cores) \
        (delayed(run_optimize)(model_class, optimizer_class, \
                               save_traj=save_traj, verbose=debug) \
                                for _ in range(sample_size))


    # results = [(run_optimize)(model_class, optimizer_class)]

    last_x = [res[0] for res in results]
    losses = [res[1] for res in results]
    list_y_traj = [res[3] for res in results]

    # find one non-converging trajectory?
    if save_traj:
        y_traj = results[0][3]
        y_traj = [np.linalg.norm(traj, axis=1) for traj in y_traj]
        plt.plot(y_traj)
        plt.show()
        from IPython import embed
        embed() or exit(0)
        for res in results:
            x_traj = res[2]
            if len(x_traj) > 1e4:
                from IPython import embed
                embed() or exit(0)

    # losses = [y_traj[-1] for y_traj in list_y_traj]
    mean_loss = np.mean(losses)

    # last_x = [x_traj[-1] for x_traj in list_x_traj]
    num_succ = [np.min(np.linalg.norm(x_end - optimum, axis=1)) < tol for x_end in last_x]

    return {
        # 'list_x_traj': list_x_traj,
        'list_y_traj': list_y_traj[:100],
        'last_x': last_x,
        # 'losses': losses,
        'mean_loss': mean_loss,
        # 'num_succ': num_succ,
        'success_rate': sum(num_succ)/sample_size,
        # 'seed_value': seed_value,
        'total_time': time.time() - start_time
    }

