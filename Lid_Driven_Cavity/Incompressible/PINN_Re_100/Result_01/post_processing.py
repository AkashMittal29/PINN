# Post Processing

#%% Loading model 
import torch
import matplotlib.pyplot as plt
import numpy
from model_class_01 import NN_Model

model = torch.load('model_01.pth', weights_only=False)


#%% Plot loss history
fid = open('loss_history.out','r')
data_str = fid.readlines()
fid.close()
data = []
for value in data_str:
    data.append(float(value))
del data_str, fid, value

fig = plt.figure(figsize=(10,5))
plt.plot(data)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Loss function history')
plt.show()


#%% Plot contours
x = torch.linspace(0, 1, 200)
y = torch.linspace(0, 1, 200)
X, Y = torch.meshgrid(x, y)

xcol = X.reshape(-1,1)
ycol = Y.reshape(-1,1)

xy = torch.cat((xcol, ycol), axis=1)

pred = model(xy)
u = pred[:,0:1]
v = pred[:,1:2]
p = pred[:,2:3]

u = u.reshape(x.numel(), y.numel())
v = v.reshape(x.numel(), y.numel())
p = p.reshape(x.numel(), y.numel())

xnp = X.numpy()
ynp = Y.numpy()
unp = u.detach().numpy()
vnp = v.detach().numpy()
pnp = p.detach().numpy()

var = vnp
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(1,1,1)
cp = plt.contourf(xnp.T, ynp.T, var.T, levels=20, cmap='rainbow')
plt.colorbar(cp)
plt.clabel(cp, inline=True, fontsize=8)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_title(r'Solution $v(x,y)$', fontsize=20)
ax.axis('equal')
plt.show()


#%% Saving data into files
numpy.savetxt("X.txt", X.numpy(), fmt="%15.10f")
numpy.savetxt("Y.txt", Y.numpy(), fmt="%15.10f")
numpy.savetxt("u.txt", u.detach().numpy(), fmt="%15.10f")
numpy.savetxt("v.txt", v.detach().numpy(), fmt="%15.10f")
numpy.savetxt("p.txt", p.detach().numpy(), fmt="%15.10f")

















