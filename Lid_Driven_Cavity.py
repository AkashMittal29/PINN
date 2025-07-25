# Solving 2D, steady, incompressible Lid Driven Cavity with Newtonian fluid using PINN
# For this, the physics (including boundary condiiton) is independent of the 
# absolute value of the pressure and is only dependent on its gradient. 

# LBFGS (Limited-Memory Broyden-Fletcher-Goldfarb-Shanno) optimization is a 
# Quasi-Newton method. It approximates the inverse Hessian matrix.
# It requires the entire dataset to find the optimized parameters. Hence, minibatches
# will not work for LBFGS. Hence, forst train with ADAM optimizer with minibatches,
# and then use LBFGS optimization to fine tune the model. 


#%% Loding Modules
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys # for sys.exit() to stop the program
import numpy
from functools import partial
import time

from mod_domain import Domain 
from mod_domain_dataloader import DomainDataLoader

# Setting random seed for CPU
torch.manual_seed(10)

import os
result_dir = 'Result_01'
os.mkdir(result_dir)


#%% Domain Definition 
domain = Domain(n_collocation=20000, n_boundaries=4, x_lim=[0,1], y_lim=[0,1])
domain.set_boundary(name='left',   boundary_type='line', 
                    n_points=500, line_end_points=[0.0, 0.0, 0.0, 1.0])
domain.set_boundary(name='bottom', boundary_type='line', 
                    n_points=500, line_end_points=[0.0, 0.0, 1.0, 0.0])
domain.set_boundary(name='right',  boundary_type='line', 
                    n_points=500, line_end_points=[1.0, 0.0, 1.0, 1.0])
domain.set_boundary(name='top',    boundary_type='line', 
                    n_points=500, line_end_points=[1.0, 1.0, 0.0, 1.0])

domain.plot_domain()
# domain.requires_grad()  # For plotting, can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
# print(Domain.__doc__)


# domain dataloader
domain.requires_grad()  # Then the batches in domain_loader will also have requires_grad True.
domain_loader = DomainDataLoader(domain, 2000, {'left':50, 'right':50, 'bottom':50, 'top':50})
# domain_batch = domain_loader.next_batch()
# print(domain_batch.collocation.requires_grad)
# domain_batch.plot_domain()  # For plotting, can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

n_batch = 10  # Batch size: consider the max. batch size among all the batches of collocation and boundaries.


#%% NN Model
# Using inbuilt layers
class NN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(2,30)) # input (:,2) -> x, y
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(30,30))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(30,30))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(30,30))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(30,30))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(30,3)) 
        self.layers.append(nn.Tanh()) # output (:,3) -> u, v, p

    def forward(self, data):
        for layer in self.layers:
            data = layer(data)
        return data

model = NN_Model()


#%% Checking NN Model
for name, param in model.named_parameters():
    print(name, param.shape)  
    
print('Wights of hidden layer 1: ', model.layers[0].weight)
print('Bias of hidden layer 1  : ', model.layers[0].bias)

a=model(torch.tensor([[0,0],[0.1,0.4]]))
print(a[:,0:1])  # 0:1 -> excluding 1 but preserves the shape.
del name, param, a


#%% Loss function
from mod_loss_function import pinn_loss


#%% Optimizer (ADAM optimizer): Adaptive Moment Estimate
# require dataloader for minibatch training.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


#%% Training loop function (for Adam optimizer)
def training_loop(domain_loader, model, loss_func, optimizer, n_batches, print_freq):
    model.train()  # setting the model to training mode. This is useful if 
                   # model has batch normalization or dropout layers. 
    for i in range(1,n_batches+1):
        domain_batch = domain_loader.next_batch()
        loss = loss_func(model, domain_batch)  # a torch scalar
        
        optimizer.zero_grad()  # To make all the stored gradients to zero.
        loss.backward()  # Backpropagation: evaluates d(loss)/d(parameter) and 
                         # stores the value in parameter.grad attribute of each parameter.
        optimizer.step() # updates parameters using the stored gradients.
        
        if i%print_freq == 0:
            loss_val = loss.item()  # converting torch scalar to python scalar. Automatically transfers to CPU.
            print(f'loss: {loss_val:>7f} [{i:>5d}/{n_batches:>5d}]')
            


#%% Training using ADAM
# Make sure that the data in domain and domain_loader has requires_grad = True
n_epochs = 20
for i in range(1,n_epochs+1):
    print(f'epoch:{i}\n---------------------------------------')
    training_loop(domain_loader, model, pinn_loss, optimizer, n_batch, 1)

print('Done');


#%% Closure function for LBFGS
def closure(model, optimizer, domain, verbose):
    """
    Computes loss and backpropagates the gradients.

    """
    global iteration, loss_history
    
    optimizer.zero_grad()
    loss = pinn_loss(model, domain)
    loss.backward()  # Backpropagation: evaluates d(loss)/d(parameter) and 
                     # stores the value in parameter.grad attribute of each parameter.
    iteration += 1
    loss_history.append(loss.item())  # converting torch scalar to python scalar. Automatically transfers to CPU.
    
    if verbose:
        print(f"iteration: {iteration}, loss: {loss.item()}")
    return loss
    
    

#%% Optimizer (LBFGS)
optimizer = torch.optim.LBFGS(model.parameters(),
                              lr = 1.0,
                              max_iter = 50000,
                              max_eval = 50000,
                              history_size = 50,
                              tolerance_grad = 1e-05,
                              tolerance_change = 0.5*numpy.finfo(float).eps,
                              line_search_fn = "strong_wolfe")


#%% Training using LBFGS
global iteration, loss_history
iteration = 0
loss_history =[]

model.train()
closure_fn = partial(closure, model, optimizer, domain, verbose=True)

time_start = time.time()
optimizer.step(closure_fn)  # Optimizing the model
elapsed = time.time()-time_start
print(f'Time elapsed = {elapsed}')


#%% Plotting loss history
fig = plt.figure(figsize=(10,5))
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Loss function history')
plt.show()


fid=open(result_dir+'/'+'loss_history.out','w')
for i in loss_history:
    fid.write(f'{i:<15.10f}\n')
fid.close()


#%% Saving model
torch.save(model, result_dir+'/'+'model_01.pth')  # Saves the entire model
# model_01 = torch.load(result_dir+'/'+'model_01.pth', weights_only=False)

# or save only the weights and load the weights to an existing instance of the same NN_Model class.
torch.save(model.state_dict(), result_dir+'/'+'model_weights_01.pth')
# model_02 = NN_Model()
# model_02.load_state_dict(torch.load(result_dir+'/'+'model_weights_01.pth', weights_only=True))

# In both cases, NN_Model class should be available.




