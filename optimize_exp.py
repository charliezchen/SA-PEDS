import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, writers

from torch.optim import SGD

from tqdm import tqdm
import json
from datetime import datetime

from utils import *
from algo import *

# python3 optimize_exp.py -m 1000 --alpha 1000 --sample-size 100
# python3 optimize_exp.py -m 1000 --alpha 0 --sample-size 100 --alpha-inc 10

args = parse_args()

N = 100
# m = 2
# lr = 1e-4
# alpha = 1
# sample_size=int(1e2)

# model_class = partial(Rastrigin,m=m, A=3)
# optimizer_class = partial(SGD, lr=lr)

# mc = MultipleCopy(model_class, optimizer_class, N)

# x_traj, y_traj = run_optimize(mc, verbose=True)


# gen_mc = lambda : MultipleCopy(model_class, optimizer_class, N)

model_class = partial(Rastrigin, N=N, m=args.m, A=3, 
                      alpha=args.alpha, alpha_inc=args.alpha_inc)
optimizer_class = partial(SGD, lr=args.lr)

result = experiment(model_class, optimizer_class, args.sample_size, 
                    np.array([0 for _ in range(args.m)]),
                    save_traj=args.debug, debug=args.debug)

print(result['success_rate'])
print(np.mean(result['mean_loss']))
