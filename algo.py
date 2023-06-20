import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, writers


class Rastrigin(nn.Module):
    upper, lower = 4, -4

    def __init__(self, m, A):
        super(Rastrigin, self).__init__()
        self.x = torch.rand(m, requires_grad=True)
        with torch.no_grad():
            self.x = self.lower + self.x * (self.upper - self.lower)
        
        self.A = A
        
    def forward(self):
        return torch.sum(self.A + self.x ** 2 - self.A * torch.cos(2 * torch.pi * self.x))

class MultipleCopy:

    def __init__(self, model_class, optimizer_class, N):
        self.models = []
        self.optimizers = []
        self.N = N

        for _ in range(N):
            model = model_class()
            self.models.append(model)
            self.optimizers.append(optimizer_class(model.parameters()))
    
    def forward(self):
        for model in self.models:
            model()
    
    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()





class PEDS_SGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
            del kwargs['alpha']
        else:
            self.alpha = 1
        super(PEDS_SGD, self).__init__(*args, **kwargs)
        
    
    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                param.grad.fill_(torch.mean(param.grad))
                # param.grad.fill_(0)
                N = param.numel()
                Omega1 = torch.full((N, N), 1.0/N)
                param.grad.add_(torch.matmul(self.alpha * (torch.eye(N) - Omega1), param))
        super(PEDS_SGD, self).step()

