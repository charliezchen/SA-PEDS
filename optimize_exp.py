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


test_function_class = eval(args.test_function)
model_class = partial(test_function_class, N=N, m=args.m, 
                      alpha=args.alpha, alpha_inc=args.alpha_inc,
                      independent=args.independent, shift=args.shift)


# def f(x):
#     return -np.exp(-x**2) - np.exp(np.cos(2*np.pi*x)) + 1 + np.e

# X = np.linspace(-2, 2, 100)
# y = [f(x) for x in X]
# plt.plot(X,y)
# plt.show()

# exit(0)

optimizer_class = partial(SGD, lr=args.lr)

result = experiment(model_class, optimizer_class, args.sample_size, 
                    np.array([args.shift for _ in range(args.m)]),
                    save_traj=args.debug, debug=args.debug)
from IPython import embed
embed() or exit(0)
print(result['success_rate'])
print(np.mean(result['mean_loss']))

