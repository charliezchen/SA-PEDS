import torch
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import argparse

import argparse
import argparse

import argparse
from collections import defaultdict

f = lambda x: -20 * np.exp(-0.2*np.sqrt(0.5*x**2)) - np.exp(0.5*(np.cos(2*np.pi*x))) + np.e + 20
x = np.linspace(-5, 5, 200)
plt.plot(x, f(x))
plt.show()
exit(0)
def group_args(args):
    groups = defaultdict(dict)
    for arg, val in vars(args).items():
        parts = arg.split('_')
        if len(parts) > 1:
            groups[parts[0]][parts[1]] = val
        else:
            groups[arg] = val
    return groups

def main():
    parser = argparse.ArgumentParser()
    
    group1 = parser.add_argument_group('group1')
    group2 = parser.add_argument_group('group2')
    
    group1.add_argument("--param1", "--group1_param1", type=int, default=1, help="Param1 in Group1")
    group1.add_argument("-b", "--group1_param2", type=int, default=2, help="Param2 in Group1")
    
    group2.add_argument("-c", "--group2_param1", type=str, default='a', help="Param1 in Group2")
    group2.add_argument("-d", "--group2_param2", type=str, default='b', help="Param2 in Group2")
    
    args = parser.parse_args()

    grouped_args = group_args(args)
    print(grouped_args)

if __name__ == "__main__":
    main()


exit(0)

# Generate x, y coordinates
lower, upper = -100, 100
samples = 200
x = np.linspace(lower, upper, samples)
# y = np.linspace(lower, upper, samples)
shift=4

# # Create 2D grid of coordinates
# X, Y = np.meshgrid(x, y)

# Define function f(x, y)
sigma = 100
f = lambda x: 3 + (x-shift)**2 - 3 * np.cos(2 * np.pi * (x-shift))
a, b, c = 20, 0.2, 2*np.pi
f = lambda x: -a*np.exp(-b*(x-shift)**2) - np.exp(np.cos(c*(x-shift)))
gauss = lambda x,y: 1/ np.sqrt(2 * np.pi) / sigma * np.exp(-1/2 * ((y-x)/sigma)**2)

# print(x[1] - x[0])
# exit(0)
# y = f(x)

# y=gauss(x,2)

# plt.plot(x,y)
# plt.show()
# exit(0)

t = np.linspace(10*lower, 10*upper, 10*samples)
delta_t = t[1] - t[0]

y = []
for i in x:
    integrant = lambda x: (f(x) * gauss(x, i))
    integral = integrant(t).sum() * delta_t


    y.append(integral)
plt.plot(x, y)
plt.show()
exit(0)
from IPython import embed
embed() or exit(0)

points = gauss(x)
y = f()




exit(0)

m = 10000
N = 100
X = np.random.uniform(0, 1, size=(m, N)).mean(axis=1)
plt.hist(X)
plt.show()

exit(0)


# Enable the profiler
with torch.profiler.profile(record_shapes=True) as prof:
    # Here goes your code
    # For example
    x = torch.randn((10, 10))
    y = torch.randn((10, 10))
    z = x @ y

# Print profiler results
print(prof.key_averages().table(sort_by="cpu_time_total"))

import pickle

with open('experiment_0620_2324.pkl', 'rb') as f:
    data = pickle.load(f)

from IPython import embed
embed() or exit(0)