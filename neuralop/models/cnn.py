import torch.nn as nn
import torch.nn.functional as F
from functools import partialmethod
import torch

class Lifting(nn.Module):
    def __init__(self, in_channels, out_channels, n_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc = Conv(in_channels, out_channels, 1)

    def forward(self, x):
        return self.fc(x)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, n_dim=2, non_linearity=F.gelu):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels 
        self.non_linearity = non_linearity
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        return x



class CNN(nn.Module):
    def __init__(self, hidden_channels, n_dim = 2,
                 in_channels=3, 
                 out_channels=1,
                 lifting_channels=256,
                 projection_channels=256,
                 n_layers=4,
                 non_linearity=F.gelu):
        super().__init__()
        self.n_dim = n_dim
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        
        self.cov1 = nn.Conv2d(self.hidden_channels, self.hidden_channels,kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.cov2 = nn.Conv2d(self.hidden_channels, 2*self.hidden_channels, kernel_size=3, padding=1)
        self.cov3 = nn.Conv2d(2*self.hidden_channels, 2*self.hidden_channels, kernel_size=3, padding=1)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.cov4 = nn.Conv2d(2*self.hidden_channels, 4*self.hidden_channels, kernel_size=3, padding=1)
        self.cov5 = nn.Conv2d(4*self.hidden_channels, 4*self.hidden_channels, kernel_size=3, padding=1)
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.cov6 = nn.Conv2d(4*self.hidden_channels, 8*self.hidden_channels, kernel_size=3, padding=1)
        self.cov7 = nn.Conv2d(8*self.hidden_channels, 8*self.hidden_channels, kernel_size=3, padding=1)

        self.lifting = Lifting(in_channels=in_channels, out_channels=self.hidden_channels, n_dim=self.n_dim)
        self.projection = Projection(in_channels=8*self.hidden_channels, out_channels=out_channels, hidden_channels=projection_channels,
                                     non_linearity=non_linearity, n_dim=self.n_dim)

    def forward(self, x):

        x = self.lifting(x)


        x = self.non_linearity(self.cov1(x))
        x = self.pool(x)
        
        x = self.non_linearity(self.cov2(x))
        x = self.non_linearity(self.cov3(x))
        x = self.pool(x)
        
        x = self.non_linearity(self.cov4(x))
        x = self.non_linearity(self.cov5(x))
        x = self.pool(x)
        
        x = self.non_linearity(self.cov6(x))
        x = self.non_linearity(self.cov7(x))


        x = self.projection(x)

        return x

