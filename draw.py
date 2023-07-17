import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re

import numpy as np
import matplotlib.pyplot as plt

# Generate x, y coordinates
lower, upper = -4, 8
samples = 200
x = np.linspace(lower, upper, samples)
y = np.linspace(lower, upper, samples)
shift=4

# Create 2D grid of coordinates
X, Y = np.meshgrid(x, y)

# Define function f(x, y)
f = lambda x: 3 + (x-shift)**2 - 3 * np.cos(2 * np.pi * (x-shift))
Z = f(X) + f(Y)

hist_file = 'debug_run.pkl'
with open(hist_file, 'rb') as f:
    hist = pickle.load(f)

one_traj = hist['list_x_traj'][0]


# Create color map using the function values
plt.imshow(Z, extent=[lower, upper, lower, upper], origin='lower', cmap='viridis', alpha=1.0)


N = one_traj[0].shape[0]
l = len(one_traj)
particles = [[] for _ in range(N)]


dot = None
print("Total length of the path:", l)
print(hist['success_rate'])

frame = min(l, 1000)
one_traj = one_traj[::int(l/frame)][:frame]


print(one_traj[-1])
# exit(0)


# GIF

for i in range(frame):
    # Clear the last point
    t = one_traj[i]

    if len(t.shape) < 2:
        t = [t]

    t1 = [i[0] for i in t]
    t2 = [i[1] for i in t]

    if dot is not None:
        dot.remove()
    dot = plt.scatter(t1, t2, color='red')

    # plt.savefig(f'frames/frame_{i:04d}.png')


    plt.pause(0.01)  # Pause to create animation effect




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
