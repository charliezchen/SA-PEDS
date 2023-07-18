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

from algo import *

import os
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Optimize using PEDS method')

    parser.add_argument("--yaml_config_path", type=str, required=True)
    parser.add_argument("--independent", action='store_true')

    parser.add_argument("--test_function", type=str, required=True)
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--m", type=int)

    parser.add_argument("--debug", action='store_true')
    return parser.parse_args()

args = parse_args()

if not os.path.exists(args.folder):
    os.mkdir(args.folder)

with open(args.yaml_config_path, "r") as infile:
    yaml_config = yaml.full_load(infile)


def run(N, m, alpha, alpha_inc, rv, test_function, independent,
        folder,
        sample_size, shift, lr, momentum, seed,
        debug):

    record = {}

    filename = f'N_{N}_m_{m}_alpha_{alpha}_inc_{alpha_inc}.pkl'
    subfolder = f"{test_function}_indep_{independent}"
    concat_folder = os.path.join(folder, subfolder)
    if not os.path.exists(concat_folder):
        os.mkdir(concat_folder)

    file_path = os.path.join(concat_folder, filename)


    # Add the meta information to the run
    record.update(vars(args))

    test_function_class = eval(test_function)
    model_class = partial(test_function_class, N=N, m=m,
                            alpha=alpha, alpha_inc=alpha_inc, 
                            shift=shift,
                            independent=independent, rv=rv)
    optimizer_class = partial(torch.optim.Adam, lr=lr)

    result = experiment(model_class, optimizer_class, 
                        sample_size,
                        seed_value=seed, debug=debug)

    record.update(result)

    if debug:
        file_path = 'debug_run.pkl'
        print("Success rate:", result['success_rate'])


    # Use 'with open' to ensure the file gets closed after writing
    # Save data to a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(record, f)

    # with open('data.pkl', 'rb') as f:
    #     data = pickle.load(f)

def run_exp(config):
    list_key = []
    list_length = []
    base_config = {}

    for k, v in config.items():
        if isinstance(v, list):
            list_key.append(k)
            list_length.append(len(v))
        else:
            base_config[k] = v
    
    cum_length = [1]
    for length in list_length:
        cum_length.append(cum_length[-1] * length)

    if not list_length:
        run(**base_config)
        exit(0)

    for i in range(np.prod(list_length)):
        con = base_config.copy()

        for j in range(len(list_key)):
            index = i // cum_length[j] % list_length[j]
            key = list_key[j]
            con[key] = config[key][index]
        run(**con)


for k, v in vars(args).items():
    if v is not None:
        yaml_config[k] = v
yaml_config.pop('yaml_config_path')

run_exp(yaml_config)
# print(yaml_config)


    

    



