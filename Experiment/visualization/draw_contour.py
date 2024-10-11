#
# ! 等高线图的绘制
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

c=9

x = np.linspace(0,1,500)  # epison
y = np.linspace(0,1,500)  # p
X, Y = np.meshgrid(x,y)
# Z = 1 - Y + X * (2* Y-1) # 分对的
Z = Y+X*(1-2*Y)/(c-1) # 分错的
L = -(1-Z)**2*np.log(Z)

plt.figure(figsize=(8,6))
contour = plt.contour(X, Y, L, levels=20,cmap='coolwarm')

cbar=plt.colorbar(contour)
cbar.set_label('L value')
plt.clabel(contour, inline=True, fontsize=8)

plt.xlim(0,0.2)
plt.ylim(0,0.2)

plt.title('false loss')
plt.xlabel('epison', fontsize=12)
plt.ylabel('p', fontsize=12)

plt.tight_layout()
plt.show()