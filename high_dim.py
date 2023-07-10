import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


N = 10
shift = 0 # Changing this may need more other changes
origin = np.ones(N) * 2


d1 = np.random.randn(N)
d1 = shift - origin
# d1 = np.ones(N)
d1 /= np.linalg.norm(d1, ord=2)
d2 = np.random.randn(N)
d2 -= np.dot(d1, d2) / np.dot(d1, d1) * d1 # Gram-Schmidt
d2 /= np.linalg.norm(d2, ord=2)


# d1 = np.array([1] + [0 for _ in range(9)])
# d2 = np.array([0 for _ in range(9)] + [1])

assert np.abs(np.dot(d1, d2)) < 1e-5, "They should be independent" + str(np.dot(d1, d2))

def f(X, A=3):
    X -= shift
    first = (A * X**2 - A * np.cos(2 * np.pi * X)).sum(axis=-1)
    second = 0 * np.dot(X-X.mean(), X - X.mean())
    return first + second

def two_f(x, y):
    X = x * d1 + y * d2 + origin
    return f(X)


lower, upper, samples = -4, 4, 400

x = np.linspace(lower, upper, samples)
y = np.linspace(lower, upper, samples)

X, Y = np.meshgrid(x, y)

Z = np.zeros_like(X)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i][j] = two_f(X[i][j], Y[i][j])

# Plot surface

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()




# plt.imshow(Z, extent=[lower, upper, lower, upper])
# plt.colorbar(label='f(x,y)')

# plt.show()