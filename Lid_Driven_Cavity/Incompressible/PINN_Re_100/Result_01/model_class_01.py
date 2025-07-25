from torch import nn

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
        self.layers.append(nn.Linear(30,3)) # output (:,3) -> u, v, p
        self.layers.append(nn.Tanh())

    def forward(self, data):
        for layer in self.layers:
            data = layer(data)
        return data
