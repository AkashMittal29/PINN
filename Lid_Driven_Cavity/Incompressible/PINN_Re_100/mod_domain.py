# Domain module

#%% 
import torch
import matplotlib.pyplot as plt
import sys # for sys.exit() to stop the program

torch.manual_seed(10)

class Domain:
    """
    Attributes
    ------------------
        collocation    : tensor[:,2], collocation points.
        n_boundaries   : scalar, number of boundaries.
                         May include body.
        x_lim          : domain's x limits [x1,x2].
        y_lim          : domain's y limits [y1,y2].
        boundary       : dictionary{name: tensor[:,2]}, 
                         boundary points.
        fig            : figure, use ax = fig.axes[0]
                         to get the first axes object.
    
    Methods
    ------------------
        set_boundary
        requires_grad
        plot_domain
        
    Note
    ------------------
    If gradient is to be computed with respect to the 
    domain tensors, then call requires_grad method to 
    make their .requires_grad = True.
    
    """
    
    def __init__(self, n_collocation, n_boundaries, x_lim, y_lim):
        x = torch.zeros((n_collocation,1))
        y = torch.zeros((n_collocation,1))
        self.collocation = torch.cat((x,y), axis=1)
        self.n_boundaries = n_boundaries
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.__set_collocation()
        self.boundary = {} # Dictionary of boundaries
        
        
    def __set_collocation(self):
        self.collocation[:,0] = self.x_lim[0]+(self.x_lim[1]-self.x_lim[0])*torch.rand_like(self.collocation[:,0])
        self.collocation[:,1] = self.y_lim[0]+(self.y_lim[1]-self.y_lim[0])*torch.rand_like(self.collocation[:,1])
        
        
    def set_boundary(self, name, boundary_type, 
                     n_points=None, line_end_points=None, # for type = 'line' 
                     file=None,                           # for type = 'curve'
                    ):
        boundary_type = boundary_type.lower()
        if len(self.boundary)<self.n_boundaries:
            if(boundary_type=='line'):
                if(n_points==None or line_end_points==None):
                    print('Error: For line boundary, n_points or line_end_points is missing')
                    sys.exit()
                
                x1, y1, x2, y2 = line_end_points
                
                if(x1==x2 and y1==y2):
                    print('Error: in boundary, line_end_points are identical.')
                    sys.exit()
                
                if(x1!=x2):
                    x = x1+torch.rand(n_points,1)*(x2-x1)
                else:
                    x = torch.zeros(n_points,1)
                    x[:] = x1
                if(y1!=y2):
                    y = y1+torch.rand(n_points,1)*(y2-y1)
                else:
                    y = torch.zeros(n_points,1)
                    y[:] = y1
                
                self.boundary[name] = torch.cat((x,y), axis=1)
                
                
            elif(boundary_type=='curve'):
                # read from the file
                print('Reading from file...NOT YET IMPLEMENTED.')
                sys.exit()
                
            else:
                 print('Error: boundary_type can be \'line\', \'curve\'.')
                 sys.exit()
            
        else:
            print(f'Error: Number of boundaries was set as \
                  {self.n_boundaries}, but more boundaries are being set.')
            sys.exit()
            
            
    def requires_grad(self):
        self.collocation.requires_grad = True
        for bound in self.boundary:
            self.boundary[bound].requires_grad = True
        
        
    def plot_domain(self):
        self.fig = plt.figure()  # ax = fig.axes[0]  # To get the first (and usually only) Axes object
        ax = self.fig.add_subplot(1,1,1)
        ax.plot(self.collocation[:,0], self.collocation[:,1], '.k')
        for bound in self.boundary:
            ax.plot(self.boundary[bound][:,0], self.boundary[bound][:,1], '.r')
                          
        