import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, writers
from torch.distributions import MultivariateNormal


class PEDS_Model(nn.Module):
    upper, lower = -5, 5
    init_noise = 30

    def __init__(self, N, m, alpha, alpha_inc, shift=0, independent=False, rv=False):
        super(PEDS_Model, self).__init__()
        center = (self.lower + torch.rand(m) * (self.upper - self.lower))
        # mvn = MultivariateNormal(torch.zeros(m), self.init_noise * torch.eye(m))

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
            # TODO: should this scale with N?
            noise = torch.randn(self.N, self.m) * self.var
            self.x = self.center + noise

        # raise NotImplemented
    
    def peds_step(self):
        if self.rv:
            self.last_center = self.center.clone()
            with torch.no_grad():
                # Rastrigin
                # Just divide by N
                # Set the thres to be 1

                # Ackley
                # Use target as 1 and set the thresh as 2
                # But it is using max. That's strange.
                #

                # self.center.grad /= self.N
                # target = 1
                # self.center.grad *= target/self.center.grad.max()
                print(self.center.data, end=' ')
                print(self.center.grad, end=' ')
                print(self.center.grad.norm(p=2))
            if torch.norm(self.center.grad, p=2) < 2:
                print("shrink var", end=' ')
                print(self.var)
            self.var -= self.alpha_inc # 1e-2
            # if self.var < 0:
            #     self.init_noise //= 2
            #     self.var = self.init_noise
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
    
    def post_step(self, optimizer):
        # dist = torch.dist(self.center, self.last_center)
        # print(dist)
        # if dist < 0.01:
        #     self.var -= 0.01
        #     # Clamp var to prevent it from going negative
        #     self.var = max(self.var, 0)
        return
        sum_ratio = 0
        param_count = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'exp_avg' in state and 'exp_avg_sq' in state:
                    ratio = state['exp_avg'] / (state['exp_avg_sq'].sqrt() + 1e-8) # Add a small number to avoid division by zero
                    sum_ratio += torch.mean(ratio) # Assuming you want mean of ratio across elements in a parameter tensor
                    param_count += 1

        average_ratio = sum_ratio / param_count if param_count > 0 else 0
        print(average_ratio)
        if average_ratio.abs() > 0.3:
            self.var -= self.alpha_inc
        # if 

        return
        # adjust learning rate according to average_ratio
        self.var -= self.alpha_inc * torch.exp(-(average_ratio.abs()))  # Add a small number to avoid division by zero
        self.var = max(self.var, 0)
        print(self.var, average_ratio.abs())

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
            print("There is nan in the function output. Check it!")
            exit(0)
        return y

    def optimal(self):
        return torch.ones(self.m) * self.shift

class Rosenbrock(PEDS_Model):
    def __init__(self, *args, **kwargs):
        self.a, self.b = 1, 100
        super(Rosenbrock, self).__init__(*args, **kwargs)

    def forward(self):
        super().forward()
        assert self.m == 2 
        
        # Compensate the shift
        x = self.x - self.shift

        return (self.a - x[:, 0])**2 + self.b * (x[:, 1] - x[:, 0]**2)**2


    def optimal(self):
        return torch.tensor([self.a - self.shift, self.a**2 - self.shift])
