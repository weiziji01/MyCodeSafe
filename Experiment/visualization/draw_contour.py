#
# ! 等高线图的绘制
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

c=9

x = np.linspace(0,1,10000)
y = np.linspace(0,1,10000)
X, Y = np.meshgrid(x,y)
Z = (Y * (c - 1 + 2 * X) + X) / (c - 1)

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS)
ax.set_title('A Simple')
plt.show()