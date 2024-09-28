
import torch.nn as nn
import torch

from box import ConfigBox
import torch.optim as optim
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Base_net(nn.Module):
    def __init__(self, config:ConfigBox):
        super(Base_net, self).__init__()

        self.config = config

        layers = config.model.layers
        self.layers = layers

        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.loss_function_no_reduction = nn.MSELoss(reduction ='none')

        self.fc = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        for i in range(len(layers)-1):
            nn.init.xavier_uniform_(self.fc[i].weight.data)
            nn.init.zeros_(self.fc[i].bias.data)
        
        if config.model.inverse:
            self.lambd = nn.Parameter(torch.tensor([[0.5], [0.5], [0.1]]), requires_grad=True)

        # parameter initialization
        self.dl=self.config.particle.dl
        self.rd=self.config.particle.rd
        self.rho=self.config.particle.rho
        self.theta = np.cos(self.config.particle.theta)

        self.ds = self.dl/self.rd
        self.rds = 1.0/self.rd
        self.ml = 1/6*math.pi*self.dl**3*self.rho
        self.ms = 1/6*math.pi*self.ds**3*self.rho


        self.gamma=self.config.particle.gamma
        self.phi=0.55
        self.g=9.81*self.theta
        self.h0=self.config.particle.h0
        self.p0=self.h0*self.rho*self.g*self.phi    

        self.t_scale = self.config.geometry.t_scale

        self.z_max = self.config.geometry.z_max
        self.z_min = self.config.geometry.z_min
        self.z_center = (self.z_max+self.z_min)/2

    
    def forward(self, x,y):
        
        a = torch.cat((x,y),1)

        for i in range(len(self.layers)-2):
            
            z = self.fc[i](a)
                        
            a = self.activation(z)
            
        a = self.fc[-1](a)

        # a = torch.clamp(a, 0.01, 0.99)
    
        return a
    
    def compute_gradient_norm(self,loss):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # Retain graph for multiple backward passes
        grad_norm = torch.tensor([p.grad.norm().item() for p in self.parameters() if p.grad is not None]).mean()
        grad_max = torch.tensor([p.grad.norm().item() for p in self.parameters() if p.grad is not None]).max()
        return grad_max/(grad_norm+1e-8)