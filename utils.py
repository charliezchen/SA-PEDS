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



def run_optimize(model_class, optimizer_class, minimal_step=1e-3, maxiter=int(1e3), save_traj = False, verbose=False):
    x_traj, y_traj = [], []
    last_x, last_y = None, None

    model = model_class()
    optimizer = optimizer_class(model.parameters())

    rv = model.rv

    std = []
    SNR_ratio = []


    iter = 0
    # Optimize the objective function
    while iter < maxiter:
        # with torch.profiler.profile(record_shapes=True) as prof:
            if rv:
                x = model.center.detach().numpy().copy()
            else:
                x = model.x.detach().numpy().copy()
            # Stopping condition
            if last_x is not None:
                steps = [np.linalg.norm(x[i]-last_x[i]) for i in range(len(x))]
                if np.max(steps) < minimal_step and model.var < 0.1:
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
            model.post_step(optimizer)
            

            if save_traj:
                x_traj.append(last_x)
                y_traj.append(last_y)
                # y_traj.append(model.x.grad.detach().numpy().copy())
                std.append(model.var)
                SNR_ratio.append(model.snr)
            
            iter += 1

        # print(prof.key_averages().table(sort_by="cpu_time_total"))
        # exit(0)

    return {
        'last_x': last_x,
        'last_y': last_y,
        'x_traj': x_traj,
        'y_traj': y_traj,
        'std': std,
        'snr': SNR_ratio,
        'iter': iter
    }


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
    # optimum,
    tol=0.5,
    seed=42,
    debug=False
):
    optimum = model_class().optimal().numpy()

    start_time = time.time()

    set_seed(seed)

    # The number of threads is equalt to the number of cores
    num_cores = multiprocessing.cpu_count()

    if not debug:
        results = Parallel(n_jobs=num_cores) \
            (delayed(run_optimize)(model_class, optimizer_class, \
                                save_traj=debug, verbose=debug) \
                                    for _ in range(sample_size))
    else:
        results = [run_optimize(model_class, optimizer_class, \
                                save_traj=debug, verbose=debug)]


    # results = [(run_optimize)(model_class, optimizer_class)]
    ret_dict = {}
    for k, _ in results[0].items():
        ret_dict[k] = [res[k] for res in results]
    
    # losses = [y_traj[-1] for y_traj in list_y_traj]
    mean_loss = np.mean(ret_dict['last_y'])

    last_x = ret_dict['last_x']

    # The variable of interrest can either be 1-D (center) or 2-D (copies)
    if len(last_x[0].shape) > 1:
        num_succ = [np.min(np.linalg.norm(x_end - optimum, axis=1)) < tol for x_end in last_x]
        last_x = [np.mean(x, axis=0) for x in last_x]
    else:
        num_succ = [np.min(np.linalg.norm(x_end - optimum)) < tol for x_end in last_x]

    mean_last_x = np.stack(last_x).mean(0)

    return {
        'list_x_traj': ret_dict['x_traj'] ,
        'list_y_traj': ret_dict['y_traj'] [:100],
        'std': ret_dict['std'],
        'mean_iter': np.mean(ret_dict['iter']),
        'snr': ret_dict['snr'],
        'mean_last_x': mean_last_x,
        'last_x': last_x,
        # 'losses': losses,
        'mean_loss': mean_loss,
        # 'num_succ': num_succ,
        'success_rate': sum(num_succ)/sample_size,
        # 'seed_value': seed_value,
        'total_time': time.time() - start_time,
        'mean_time': np.mean(time.time() - start_time)

    }

