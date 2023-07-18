import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# Generate x, y coordinates
lower, upper = -10, 10
samples = 200
x = np.linspace(lower, upper, samples)
y = np.linspace(lower, upper, samples)
shift=4

# Create 2D grid of coordinates
X, Y = np.meshgrid(x, y)

# Define function f(x, y)
f = lambda x: 3 + (x-shift)**2 - 3 * np.cos(2 * np.pi * (x-shift))
Z = f(X) + f(Y)

# Ackley
# f = lambda x, y: -20*np.exp(-0.2*np.sqrt(((x-shift)**2+(y-shift)**2)/2)) - np.exp((np.cos(2*np.pi*(x-shift)) + np.cos(2*np.pi*(y-shift))/2)) + np.e + 20
# Z = f(X, Y)

# Rosenbrock
a, b = 1, 100
f = lambda x,y: (a - (x-shift))**2 + b * (y - shift - (x-shift)**2)**2
Z = f(X,Y)
print(f(1,1))
Z = np.log(Z + 1)


hist_file = 'debug_run.pkl'
with open(hist_file, 'rb') as f:
    hist = pickle.load(f)

one_traj = hist['list_x_traj'][0]
std = hist['std'][0]


# Create color map using the function values
# plt.imshow(Z, extent=[lower, upper, lower, upper], origin='lower', cmap='viridis', alpha=1.0)


N = one_traj[0].shape[0]
l = len(one_traj)
particles = [[] for _ in range(N)]


dot = None
circle = None
print("Total length of the path:", l)
print(hist['success_rate'])
print("init x:", (hist['list_x_traj'][0][0]))

print("last y:", np.mean(hist['list_y_traj'][0][-1]))
print("last std:", np.mean(hist['std'][0][-1]))

frame = min(l, 100)
one_traj = one_traj[::int(l/frame)][:frame]
std = std[::int(l/frame)][:frame]


# print(one_traj[-1])
# exit(0)



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
    circle = Circle((t1[0], t2[0]), std[i], fill=False, edgecolor='red')
    ax.add_patch(circle)


    # # Dynamically adjust the axis limits
    # min_x = min(min(t1) - 1, lower)
    # max_x = max(max(t1) + 1, upper)
    # min_y = min(min(t2) - 1, lower)
    # max_y = max(max(t2) + 1, upper)
    
    # ax.set_xlim([min_x, max_x])
    # ax.set_ylim([min_y, max_y])


    # plt.savefig(f'frames/frame_{i:04d}.png')


    plt.pause(0.01)  # Pause to create animation effect
plt.show()



# # Add colorbar
# plt.colorbar(label='f(x,y)')

# # Show the plot
# plt.show()



# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter 

# # initial setup
# fig, ax = plt.subplots()

# dot, = plt.plot([], [], 'ro') 

# def init():
#     ax.set_xlim(-10, 10)  # assuming the points are in this range
#     ax.set_ylim(-10, 10)  # change accordingly if needed
#     return dot,

# def update(i):
#     # Clear the last point
#     t = one_traj[i]
#     t1 = [i[0] for i in t]
#     t2 = [i[1] for i in t]
#     dot.set_data(t1, t2)
#     return dot,

# # Create animation
# ani = FuncAnimation(fig, update, frames=range(frame), init_func=init, blit=True)

# # Save as gif
# writer = PillowWriter(fps=20) 
# ani.save("animation.gif", writer=writer)
