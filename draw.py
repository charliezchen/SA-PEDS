import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

hist_file = 'debug_run.pkl'
with open(hist_file, 'rb') as f:
    debug_run = pickle.load(f)

model_config = debug_run['model_config']
exp_config = debug_run['experiment_config']
shift=model_config['shift']
test_function = exp_config['test_function']
result = debug_run['result']

rv = model_config['rv']
result = {k:v[0] if isinstance(v, list) else v for k,v in result.items()}
one_traj = result['list_x_traj']
std = result['std']

l = len(one_traj)
print("Total length of the path:", l)
print(result['success_rate'])
print("init x:", (result['list_x_traj'][0]))
print("last x:", (result['list_x_traj'][-1]))

if rv:
    x0 = result['list_x_traj'][0][0]
    y0 = result['list_x_traj'][0][1]
else:
    x0 = result['list_x_traj'][0][:, 0]
    y0 = result['list_x_traj'][0][:, 1]
std0 = result['std'][0] 
std0 = max(0, std0)


lower, upper = -10, 10
if rv:
    lower = min(lower, x0 - std0)
    lower = min(lower, y0 - std0)
    upper = max(upper, x0 + std0)
    upper = max(upper, y0 + std0)
else:
    lower = min(lower, np.min(x0) - std0)
    lower = min(lower, np.min(y0) - std0)
    upper = max(upper, np.max(x0) + std0)
    upper = max(upper, np.max(y0) + std0)

print("last y:", np.mean(result['list_y_traj'][-1]))

print("last std:", (result['std'][-1]))

# Generate x, y coordinates
samples = 200
x = np.linspace(lower, upper, samples)
y = np.linspace(lower, upper, samples)

# Create 2D grid of coordinates
X, Y = np.meshgrid(x, y)

# Define function f(x, y)
if test_function=='Rastrigin':
    f = lambda x: 3 + (x-shift)**2 - 3 * np.cos(2 * np.pi * (x-shift))
    Z = f(X) + f(Y)
elif test_function == 'Ackley':
    # Ackley
    f = lambda x, y: -20*np.exp(-0.2*np.sqrt(((x-shift)**2+(y-shift)**2)/2)) - np.exp((np.cos(2*np.pi*(x-shift)) + np.cos(2*np.pi*(y-shift))/2)) + np.e + 20
    Z = f(X, Y)
elif test_function == 'Rosenbrock':
    # Rosenbrock
    a, b = 1, 100
    f = lambda x,y: (a - (x-shift))**2 + b * (y - shift - (x-shift)**2)**2
    Z = f(X,Y)
    Z = np.log(Z + 1)
    # from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='viridis')
    # plt.show()
    # exit(0)

else:
    print(f"Unrecognized test function {test_function}")
    exit(0)


N = one_traj[0].shape[0]
particles = [[] for _ in range(N)]


dot = None
circle = None


# Speed up the animation by picking only part of the frames
frame = min(l, 100)
one_traj = one_traj[::int(l/frame)][:frame]
std = std[::int(l/frame)][:frame]

x_traj = result['list_y_traj']
SNR_ratio = result['snr']
# from IPython import embed
# embed() or exit(0)
SNR_ratio = [(i).numpy() for i in SNR_ratio]
fig, ax = plt.subplots()
losses = [np.mean(i) for i in x_traj]

# SNR_ratio = np.array(SNR_ratio)
# losses = np.array(losses)
# ax.plot(losses, label='loss')
# ax.plot(SNR_ratio, label='variance of gradient')
# ax.legend()
# plt.show()


# GIF
fig, ax = plt.subplots()


for i in range(frame):
    # Clear the last point
    t = one_traj[i]

    if len(t.shape) < 2:
        t = [t]

    t1 = [i[0] for i in t]
    t2 = [i[1] for i in t]

    if dot is not None:
        dot.remove()
    if circle is not None:
        circle.remove()

    plt.imshow(Z, extent=[lower, upper, lower, upper], origin='lower', cmap='viridis', alpha=1.0)

    dot = ax.scatter(t1, t2, color='red')
    if rv:
        circle = Circle((t1[0], t2[0]), std[i], fill=False, edgecolor='red')
        ax.add_patch(circle)


    # # Dynamically adjust the axis limits
    # min_x = min(min(t1) - 1, lower)
    # max_x = max(max(t1) + 1, upper)
    # min_y = min(min(t2) - 1, lower)
    # max_y = max(max(t2) + 1, upper)
    
    # ax.set_xlim([min_x, max_x])
    # ax.set_ylim([min_y, max_y])


    # Uncomment one of the following two lines
    # plt.savefig(f'frames/frame_{i:04d}.png') # if you want to save pictures
    plt.pause(0.01)  # if you want to view the animations


