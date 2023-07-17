import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, writers


class PEDS_Model(nn.Module):
    upper, lower = 100, -100
    init_noise = 10

    def __init__(self, N, m, alpha, alpha_inc, shift=0, independent=False, rv=False):
        super(PEDS_Model, self).__init__()
        center = (self.lower + torch.rand(m) * (self.upper - self.lower))
        if rv:
            self.center = nn.Parameter(center)
            self.var = self.init_noise
        else:
            self.x = nn.Parameter(center + torch.randn(N, m) * self.init_noise)

        self.shift = shift

        self.m = m
        self.N = N
        self.alpha = alpha
        self.alpha_inc = alpha_inc
        self.rv = rv


        self.independent = independent
        self.I = torch.eye(self.N) 

    
    # This must be overloaded and called in the overloaded function before anything else
    def forward(self):
        if self.rv:
            noise = torch.randn(self.N, self.m) * self.var
            self.x = self.center + noise

        # raise NotImplemented
    
    def peds_step(self):
        if self.rv:
            with torch.no_grad():
                self.center.grad /= self.N
            self.var -= self.alpha_inc # 1e-2
            self.var = max(self.var, 0)

            return 
        
        with torch.no_grad():
            if self.independent:
                projected_grad = self.x.grad
            else:
                projected_grad = torch.ones_like(self.x.grad) * torch.mean(self.x.grad, dim=0)
            
            if torch.norm(projected_grad, p=2) < 1: # TODO: TO change
                self.alpha += self.alpha_inc
            else:
                self.alpha = 0

            mean = torch.mean(self.x, dim=0)
            attraction = self.alpha * (self.x - mean)

            self.x.grad = (projected_grad + attraction) 

class Rastrigin(PEDS_Model):
    def __init__(self, *args, **kwargs):
        self.A = 3 # TODO: pass in arguments

        super(Rastrigin, self).__init__(*args, **kwargs)
    def forward(self):
        super().forward()
        return torch.sum(self.A + (self.x - self.shift) ** 2 - self.A * torch.cos(2 * torch.pi * (self.x - self.shift)), dim=-1)

class Ackley(PEDS_Model):
    def __init__(self, *args, **kwargs):
        self.a, self.b, self.c = 1, 1, 2*torch.pi # TODO: pass in arguments

        super(Ackley, self).__init__(*args, **kwargs)

    def forward(self):
        super().forward()

        # Compensate the shift
        x = self.x - self.shift

        term1 = -self.a * torch.exp(-self.b * torch.sqrt(torch.mean(x**2, dim=1)))
        term2 = -torch.exp(torch.mean(torch.cos(self.c * x), dim=1))
        y = term1 + term2 + self.a + torch.e 
        if torch.isnan(y).any():
            print("There is nan in the function output. Check it!")
            exit(0)
        return y

        