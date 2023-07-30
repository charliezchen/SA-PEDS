import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, writers
from torch.distributions import MultivariateNormal
from abc import abstractmethod


class PEDS_Model(nn.Module):
    def __init__(self, N, m, alpha, alpha_inc, upper, lower,
                 init_noise, shift=0, independent=False, rv=False):
        super(PEDS_Model, self).__init__()
        center = (lower + torch.rand(m) * (upper - lower))

        if rv:
            self.center = nn.Parameter(center)
            self.var = init_noise
        else:
            self.x = nn.Parameter(center + torch.randn(N, m) * init_noise)
            self.var = -1

        self.shift = shift

        self.m = m
        self.N = N
        self.alpha = alpha
        self.alpha_inc = alpha_inc
        self.rv = rv

        self.independent = independent
        if self.independent:
            assert not rv, "Independent is used as baseline. It cannot be SGD."
        self.I = torch.eye(self.N) 

    
    # This must be overloaded and called in the overloaded function before anything else
    @abstractmethod
    def forward(self):
        if self.rv:
            # White noise independent of dimension and particle index
            noise = torch.randn(self.N, self.m) * self.var
            self.x = self.center + noise
            # self.x.retain_grad()
    
    def peds_step(self):
        if not self.rv:
            with torch.no_grad():
                if self.independent:
                    projected_grad = self.x.grad
                    return
                else:
                    projected_grad = torch.ones_like(self.x.grad) * torch.mean(self.x.grad, dim=0)

                mean = torch.mean(self.x, dim=0)
                attraction = self.alpha * (self.x - mean)

                self.x.grad = (projected_grad + attraction) 

        if self.rv:
            self.var -= self.alpha_inc
            self.var = max(self.var, 0)
        else:
            self.alpha += self.alpha_inc
    
    # Now, this function is for debugging purpose only
    def post_step(self, optimizer):
        # return
        # dist = torch.dist(self.center, self.last_center)
        # print(dist)
        # if dist < 0.01:
        #     self.var -= 0.01
        #     # Clamp var to prevent it from going negative
        #     self.var = max(self.var, 0)
        # return
        sum_ratio = 0
        param_count = 0
        self.snr = torch.tensor(0)
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'exp_avg' in state and 'exp_avg_sq' in state:
                    ratio = state['exp_avg'] / (state['exp_avg_sq'].sqrt() + 1e-8) # Add a small number to avoid division by zero
                    # sum_ratio = ratio
                    # sum_ratio += torch.mean(ratio.abs()) # Assuming you want mean of ratio across elements in a parameter tensor
                    # self.snr = self.x.grad.var(dim=0)
                    self.snr = ratio.clone()
                    param_count += 1
        assert param_count <= 1,param_count
        # with torch.no_grad():
        #     pass
        #     self.var -= self.alpha_inc # * torch.exp(-(ratio**2).mean()).numpy()
        # print(self.var)
        # from IPython import embed
        # embed() or exit(0)
        # exit(0)
        # average_ratio = sum_ratio / param_count if param_count > 0 else 0
        # print(average_ratio)
        return 


    @abstractmethod
    def optimal(self):
        raise NotImplemented

class Rastrigin(PEDS_Model):
    def __init__(self, *args, **kwargs):
        self.A = 3 # TODO: pass in arguments

        super(Rastrigin, self).__init__(*args, **kwargs)
    def forward(self):
        super().forward()
        return torch.sum(self.A + (self.x - self.shift) ** 2 - self.A * torch.cos(2 * torch.pi * (self.x - self.shift)), dim=-1)

    def optimal(self):
        return torch.ones(self.m) * self.shift

class Ackley(PEDS_Model):
    def __init__(self, *args, **kwargs):
        self.a, self.b, self.c = 20, 0.2, 2*torch.pi # TODO: pass in arguments

        super(Ackley, self).__init__(*args, **kwargs)

    def forward(self):
        super().forward()

        # Compensate the shift
        x = self.x - self.shift

        term1 = -self.a * torch.exp(-self.b * torch.sqrt(torch.mean(x**2, dim=1)))
        term2 = -torch.exp(torch.mean(torch.cos(self.c * x), dim=1))
        y = term1 + term2 + self.a + torch.e 
        if torch.isnan(y).any():
            print("[WARNING]: There is nan in the function output. Check it!")
            # from IPython import embed
            # embed() or exit(0)
            # exit(0)
        return y

    def optimal(self):
        return torch.ones(self.m) * self.shift

class Rosenbrock(PEDS_Model):
    def __init__(self, *args, **kwargs):
        self.a, self.b = 1, 100 # TODO: pass in arguments
        super(Rosenbrock, self).__init__(*args, **kwargs)

    def forward(self):
        super().forward()
        assert self.m == 2 
        
        # Compensate the shift
        x = self.x - self.shift

        return (self.a - x[:, 0])**2 + self.b * (x[:, 1] - x[:, 0]**2)**2


    def optimal(self):
        return torch.tensor([self.a + self.shift, self.a**2 + self.shift])
