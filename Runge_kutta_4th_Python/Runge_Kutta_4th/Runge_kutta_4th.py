import numpy as np
from matplotlib import pyplot as plt

def rk4(f, h, t0, y0):
    k1 = h*f(t0, y0)
    k2 = h*f(t0 + (h/2), y0 + (k1/2))
    k3 = h*f(t0 + (h/2), y0 + (k2/2))
    k4 = h*f(t0 + h, y0 + k3)

    y1 = y0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    return np.array(y1)

def lorenz(t, y):
    dy = [sigma*(y[1] - y[0]),
          y[0]*(rho - y[2]) - y[1],
          y[0]*y[1] - beta*y[2]
    ]
    return np.array(dy)


sigma = 10
beta = 8/3
rho = 28

y0 = [-8, 8, 27]

h = 0.01
T = 10
number_of_points = int(T/h)
time = np.linspace(0, T, number_of_points)

Y = np.zeros((3, number_of_points))

yin = y0
Y[:, 0] = yin

for i in range(number_of_points - 1):
    yout = rk4(lorenz, h, time[i], yin)
    Y[:, i+1] = yout
    yin = yout

ax = plt.figure().add_subplot(projection = '3d')
ax.plot(Y[0, :], Y[1, :], Y[2, :])
plt.show()

