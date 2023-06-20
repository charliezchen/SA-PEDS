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


# optimizer = partial(torch.optim.SGD, lr=1e-4)
# result = experiment(1, rast1, optimizer, int(1e4), 0, 0.1)


alpha = 1

sample_size = int(1e4)

all_records = []

for N in range(1, 11):
    record = {}
    record['N'] = N
    record['Optim'] = 'PEDS'
    record['Sample size'] = sample_size
    record['Alpha'] = alpha
    record['Projector'] = 'Mean field'

    optimizer = partial(PEDS_SGD, alpha=alpha, lr=1e-4)
    result = experiment(N, rast1, optimizer, sample_size, 0, 0.1)
    record.update(result)
    all_records.append(record)


# Format current date and time into a string
now = datetime.now()  # get current date and time
date_time = now.strftime("%m%d_%H%M")  # format date and time as string

# Use 'with open' to ensure the file gets closed after writing
filename = 'experiment_{}.json'.format(date_time)  # create filename with date and time
with open(filename, 'w') as json_file:
    json.dump(all_records, json_file)

