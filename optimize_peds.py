import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, writers

from torch.optim import SGD

from tqdm import tqdm
import json
import pickle
from datetime import datetime

from utils import *
from algo import PEDS_SGD


args = parse_args()


all_records = []

# Format current date and time into a string
now = datetime.now()  # get current date and time
date_time = now.strftime("%m%d_%H%M")  # format date and time as string
filename = 'experiment_m{}_alphainc{}_{}.pkl'.format(args.m, args.alpha_inc, date_time)  # create filename with date and time

# Add the meta information to the run
all_records.append(vars(args))

for N in range(args.lower_N, args.upper_N + 1):
    record = {}
    record['N'] = N
    # record['Optim'] = 'PEDS'
    # record['Sample size'] = sample_size
    # record['Alpha'] = alpha
    # record['Projector'] = 'Mean field'
    # record['Notes'] = "Increase alpha with 0.1 with each step"
    # record['Rastrigin_A'] = A

    # naive = record['Optim'] == 'naive'
    model_class = partial(Rastrigin, N=N, m=args.m, A=args.A, 
                          alpha=args.alpha, alpha_inc=args.alpha_inc, naive=args.naive)
    optimizer_class = partial(SGD, lr=args.lr)

    result = experiment(model_class, optimizer_class, 
                        args.sample_size, np.array([0 for _ in range(args.m)]),
                        seed_value=args.seed)

    record.update(result)
    all_records.append(record)


    # Use 'with open' to ensure the file gets closed after writing
    # Save data to a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(all_records, f)

    # with open('data.pkl', 'rb') as f:
    #     data = pickle.load(f)


