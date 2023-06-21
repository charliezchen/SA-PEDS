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



alpha = 1
lr = 1e-4
m = 2
sample_size = int(1e4)
A=3

all_records = []

# Format current date and time into a string
now = datetime.now()  # get current date and time
date_time = now.strftime("%m%d_%H%M")  # format date and time as string


for N in range(1, 11):
    record = {}
    record['N'] = N
    # record['Optim'] = 'PEDS'
    # record['Sample size'] = sample_size
    # record['Alpha'] = alpha
    # record['Projector'] = 'Mean field'
    # record['Notes'] = "Increase alpha with 0.1 with each step"
    # record['Rastrigin_A'] = A

    # naive = record['Optim'] == 'naive'
    model_class = partial(Rastrigin, N=N, m=m, A=3, alpha=alpha)
    optimizer_class = partial(SGD, lr=lr)

    result = experiment(model_class, optimizer_class, sample_size, np.array([0 for _ in range(m)]))

    record.update(result)
    all_records.append(record)


    # Use 'with open' to ensure the file gets closed after writing
    filename = 'experiment_{}.pkl'.format(date_time)  # create filename with date and time
    # Save data to a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(all_records, f)

    # with open('data.pkl', 'rb') as f:
    #     data = pickle.load(f)


