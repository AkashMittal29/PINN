# Domain Dataloader module

#%% 
from torch.utils.data import DataLoader
from mod_domain import Domain

class DomainDataLoader:
    """
    Attributes
    ------------------
        domain    : object of class Domain.
                    It refers to the memory of the object passed.
        
    Methods
    ------------------
        set_boundary
        
    Note
    ------------------
    When the object of Domain class is returned and used
    such as for gradient calculation, then the returned object's
    tensors' requires_grad shall be manually assigned True. 
    
    """
    
    def __init__(self, domain, colloc_batch_size, boundaries_batch_size):
        """
        domain    : object of class Domain
        domain_batch_size : int; batch size of the interior collocation points.
        boundaries_batch_size : dictionary{boundary name: batch size <int>}.
        
        """
        self.domain = domain  # Passed by reference in python
        self.colloc_batch_size = colloc_batch_size
        self.boundaries_batch_size = boundaries_batch_size
        self.boundary_loader = {}
        self.__collocation_loader_method()      # Creating iterator
        for key in self.boundaries_batch_size:
            self.__boundary_loader_method(key)  # Creating iterator
            
            
    def __collocation_loader_method(self):
        self.collocation_loader = iter(DataLoader(self.domain.collocation, 
                                         batch_size = self.colloc_batch_size, 
                                         shuffle=True))  # Creating iterator
        # Since, shuffle=True in the DataLoader, when iterator is created
        # using iter(), data is shuffled by the DataLoader.
        
        
    def __boundary_loader_method(self, key):
        self.boundary_loader[key] = iter(DataLoader(self.domain.boundary[key], 
                                      batch_size = self.boundaries_batch_size[key], 
                                      shuffle=True))  # Creating iterator
        # Since, shuffle=True in the DataLoader, when iterator is created
        # using iter(), data is shuffled by the DataLoader.
        
        
    def next_batch(self):
        domain_batch = Domain(n_collocation = self.colloc_batch_size, 
                              n_boundaries  = self.domain.n_boundaries, 
                              x_lim = self.domain.x_lim, 
                              y_lim = self.domain.y_lim)
        try:
            domain_batch.collocation = next(self.collocation_loader)
        except StopIteration:  # Recreating iterator when data is exhausted in iterator.
            self.__collocation_loader_method()  
            domain_batch.collocation = next(self.collocation_loader)
        
        for key in self.boundaries_batch_size:
            try:
                domain_batch.boundary[key] = next(self.boundary_loader[key])
            except StopIteration:  # Recreating iterator when data is exhausted in iterator.
                self.__boundary_loader_method(key)
                domain_batch.boundary[key] = next(self.boundary_loader[key])
            
        return domain_batch
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    