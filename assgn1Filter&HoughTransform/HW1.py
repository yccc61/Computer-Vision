import numpy as np
import math
from matplotlib import pyplot as plt

def rho(theta, x, y):
   return x*np.sin(theta)+ y*np.cos(theta)

x = np.linspace(0, math.pi)

plt.plot(x, rho(x, 10, 10), color='red')
plt.plot(x, rho(x, 20, 20), color='blue')
plt.plot(x, rho(x, 30, 30), color='green')
plt.xlim(-0, 2*math.pi)
plt.ylim(-50,50)
plt.grid()
plt.show() 