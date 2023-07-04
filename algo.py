import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, writers


class PEDS_Model(nn.Module):
    upper, lower = 2, -2

    def __init__(self, N, m, alpha, alpha_inc, shift=0, independent=False, naive=False):
        super(PEDS_Model, self).__init__()
        self.x = nn.Parameter(self.lower + torch.rand(N, m) * (self.upper - self.lower))
        self.shift = shift

        self.m = m
        self.N = N
        self.projector = torch.full((N,N), 1/N)
        self.alpha = alpha
        self.alpha_inc = alpha_inc

        if naive:
            self.projector = torch.eye(N)
            self.alpha = 0

        self.independent = independent
        self.I = torch.eye(self.N) 

        
    def forward(self):
        raise NotImplemented
    
    def peds_step(self):
        with torch.no_grad():
            # projector = self.projector + 0.1 * torch.randn(self.projector.shape)
            projector = self.projector
            if self.independent:
                projected_grad = self.x.grad
            else:
                projected_grad = torch.matmul(projector, self.x.grad)

            mean = torch.mean(self.x, dim=0)
            attraction = self.alpha * (self.x - mean)
            # attraction = self.alpha * torch.matmul((self.I - projector), self.x)

            self.x.grad = (projected_grad + attraction)
        self.alpha += self.alpha_inc

class Rastrigin(PEDS_Model):
    def __init__(self, *args, **kwargs):
        self.A = 3 # TODO: pass in arguments

        super(Rastrigin, self).__init__(*args, **kwargs)
    def forward(self):
        return torch.sum(self.A + (self.x - self.shift) ** 2 - self.A * torch.cos(2 * torch.pi * (self.x - self.shift)), dim=-1)

class Ackley(PEDS_Model):
    def __init__(self, *args, **kwargs):
        self.a, self.b, self.c = 1, 1, 2*torch.pi # TODO: pass in arguments

        super(Ackley, self).__init__(*args, **kwargs)

    def forward(self):
        term1 = -self.a * torch.exp(-self.b * torch.sqrt(torch.mean(self.x**2, dim=1)))
        term2 = -torch.exp(torch.mean(torch.cos(self.c * self.x), dim=1))
        y = term1 + term2 + self.a + torch.e 
        if torch.isnan(y).any():
            print("There is nan in the function output. Check it!")
            exit(0)
        return y

        