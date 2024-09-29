#This function corresponded to question 2.4 on the write up
import numpy as np
import matplotlib.pyplot as plt

def hough(theta, p):
   return p*np.cos(theta)+p*np.sin(theta)

x = np.linspace(0, np.pi)

plt.plot(x, hough(x,10), color='red')
plt.plot(x, hough(x,20), color='blue')
plt.plot(x, hough(x,30), color='green')

plt.title('Hough Transform')  
plt.grid(True)
plt.show()