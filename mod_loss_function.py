# Loss Function module 

#%%
import torch

# Defining global constants
global RE_REF
RE_REF = 100.0  # Reference Reynolds Number

def pinn_loss(model, domain):
    """
    Computes loss for PINN.
    
    Parameters
    ----------
    model  : PINN model; input (:,2)->(x,y); output (:,3)->(u,v,p)
    domain : class Domain
             Tensors for all the interior and boundary 
             points with requires_grad shall be True.

    Returns
    -------
    loss   : scalar, real

    """
    
    # TODO : before calling loss, make all gradients zero.
    
    global RE_REF
    RE_REF_inv = 1/RE_REF
    
    xy = domain.collocation
    pred = model(xy)  # Prediction
    u = pred[:,0:1]  # (:,1)
    v = pred[:,1:2]
    p = pred[:,2:3]
    
    # Governing equation loss
    # rho = 1, density, being incompressible and rho is dimensionless
    # mu  = 1, dynamic viscosity, constant vicosity and mu is dimensionless
    # non-dimensionalized using the reference density and dynamic viscosity.
    du_dxy = torch.autograd.grad(u,xy,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
    du_dx  = du_dxy[:,0:1]
    du_dy  = du_dxy[:,1:2]
    dv_dxy = torch.autograd.grad(v,xy,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
    dv_dx  = dv_dxy[:,0:1]
    dv_dy  = dv_dxy[:,1:2]
    
    d2u_dx2 = torch.autograd.grad(du_dx,xy,grad_outputs=torch.ones_like(du_dx),retain_graph=True,create_graph=True)[0][:,0:1]
    d2u_dy2 = torch.autograd.grad(du_dy,xy,grad_outputs=torch.ones_like(du_dy),retain_graph=True,create_graph=True)[0][:,1:2]
    d2v_dx2 = torch.autograd.grad(dv_dx,xy,grad_outputs=torch.ones_like(dv_dx),retain_graph=True,create_graph=True)[0][:,0:1]
    d2v_dy2 = torch.autograd.grad(dv_dy,xy,grad_outputs=torch.ones_like(dv_dy),retain_graph=True,create_graph=True)[0][:,1:2]   
    
    u2 = u*u
    v2 = v*v
    uv = u*v
    
    du2_dx = torch.autograd.grad(u2,xy,grad_outputs=torch.ones_like(u2),retain_graph=True,create_graph=True)[0][:,0:1]
    dv2_dy = torch.autograd.grad(v2,xy,grad_outputs=torch.ones_like(v2),retain_graph=True,create_graph=True)[0][:,1:2]
    duv_dxy = torch.autograd.grad(uv,xy,grad_outputs=torch.ones_like(uv),retain_graph=True,create_graph=True)[0]
    duv_dx = duv_dxy[:,0:1]
    duv_dy = duv_dxy[:,1:2]
    
    dp_dxy = torch.autograd.grad(p,xy,grad_outputs=torch.ones_like(p),retain_graph=True,create_graph=True)[0]
    dp_dx = dp_dxy[:,0:1]
    dp_dy = dp_dxy[:,1:2]
    
    l1_pde = du_dx + dv_dy                                             # Continuity equation loss
    l2_pde = du2_dx + duv_dy + dp_dx - RE_REF_inv*(d2u_dx2 + d2u_dy2)  # x-momentum
    l3_pde = duv_dx + dv2_dy + dp_dy - RE_REF_inv*(d2v_dx2 + d2v_dy2)  # y-momentum
    
    # Top Boundary Condition loss
    xy = domain.boundary['top']
    pred = model(xy)
    u = pred[:,0:1]  # (:,1)
    v = pred[:,1:2]
    p = pred[:,2:3]
    dp_dy = torch.autograd.grad(p,xy,grad_outputs=torch.ones_like(p),retain_graph=True,create_graph=True)[0][:,1:2]
    l1 = u-1   # u_top-1 = 0
    l2 = v     # v_top = 0
    l3 = dp_dy # dp_dy = 0
    
    # Right Boundary Condition loss
    xy = domain.boundary['right']
    pred = model(xy)
    u = pred[:,0:1]  # (:,1)
    v = pred[:,1:2]
    p = pred[:,2:3]
    dp_dx = torch.autograd.grad(p,xy,grad_outputs=torch.ones_like(p),retain_graph=True,create_graph=True)[0][:,0:1]
    l4 = u     # u_right-1 = 0
    l5 = v     # v_right = 0
    l6 = dp_dx # dp_dy = 0
    
    # Bottom Boundary Condition loss
    xy = domain.boundary['bottom']
    pred = model(xy)
    u = pred[:,0:1]  # (:,1)
    v = pred[:,1:2]
    p = pred[:,2:3]
    dp_dy = torch.autograd.grad(p,xy,grad_outputs=torch.ones_like(p),retain_graph=True,create_graph=True)[0][:,1:2]
    l7 = u     # u_bottom-1 = 0
    l8 = v     # v_bottom = 0
    l9 = dp_dy # dp_dy = 0
    
    # Left Boundary Condition loss
    xy = domain.boundary['left']
    pred = model(xy)
    u = pred[:,0:1]  # (:,1)
    v = pred[:,1:2]
    p = pred[:,2:3]
    dp_dx = torch.autograd.grad(p,xy,grad_outputs=torch.ones_like(p),retain_graph=True,create_graph=True)[0][:,0:1]
    l10 = u     # u_left-1 = 0
    l11 = v     # v_left = 0
    l12 = dp_dx # dp_dy = 0
    
    loss = 200*torch.mean(l1_pde**2) + 100*torch.mean(l2_pde**2) + 100*torch.mean(l3_pde**2) + \
           10*torch.mean(l1**2) + 10*torch.mean(l2**2) + 10*torch.mean(l3**2) + \
           100*torch.mean(l4**2) + 100*torch.mean(l5**2) + 100*torch.mean(l6**2) + \
           10*torch.mean(l7**2) + 10*torch.mean(l8**2) + 10*torch.mean(l9**2) + \
           100*torch.mean(l10**2) + 100*torch.mean(l11**2) + 100*torch.mean(l12**2) 
           
    return loss
    





