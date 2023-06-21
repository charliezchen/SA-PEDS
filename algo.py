import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, writers


class Rastrigin(nn.Module):
    upper, lower = 2, -2

    def __init__(self, N, m, A, alpha, naive=False):
        super(Rastrigin, self).__init__()
        self.x = nn.Parameter(self.lower + torch.rand(N, m) * (self.upper - self.lower))
        self.A = A

        self.N = N
        self.projector = torch.full((N,N), 1/N)
        self.alpha = alpha

        if naive:
            self.projector = torch.eye(N)
            self.alpha = 0


        self.I = torch.eye(self.N) 

        
    def forward(self):
        return torch.sum(self.A + self.x ** 2 - self.A * torch.cos(2 * torch.pi * self.x), dim=-1)
    
    def peds_step(self):
        with torch.no_grad():
            # projector = self.projector + 0.1 * torch.randn(self.projector.shape)
            projector = self.projector
            projected_grad = torch.matmul(projector, self.x.grad)
            attraction = self.alpha * torch.matmul((self.I - projector), self.x)

            self.x.grad = (projected_grad + attraction)
        self.alpha += .1
    

class MultipleCopy:

    def __init__(self, model_class, optimizer_class, N):
        self.models = []
        self.optimizers = []
        # PEDS is disabled essentially
        # self.projector = torch.eye(N)
        

        for _ in range(N):
            model = model_class()
            self.models.append(model)
            self.optimizers.append(optimizer_class(model.parameters()))
    
    def forward(self):
        y = []
        for model in self.models:
            y.append(model.forward())
        return y
    def get_x(self):
        x = []
        for model in self.models:
            x.append(model.x.detach().numpy().copy())
        
        return x
    
    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()
    
    def peds_step(self):
        
        # This is a matrix of shape N x m
        grad = torch.stack([m.x.grad.detach() for m in self.models])
        projected_grad = torch.matmul(self.projector, grad)
        # for model in self.models:
        #     model.x.grad.set_(projected_grad[])
        
        param = torch.stack([m.x.data.detach() for m in self.models])
        attraction = -self.alpha * torch.matmul((torch.eye(self.N) - self.projector), param)

        sum_of_grad = projected_grad + attraction

        for index in range(self.N):
            self.models[index].x.grad.set_(sum_of_grad[index, :])
        
        self.step()
        # from IPython import embed
        # embed() or exit(0)


    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    





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

