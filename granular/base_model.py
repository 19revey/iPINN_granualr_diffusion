
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

        self.z_upper_limit = self.config.geometry.z_upper_limit
        self.z_lower_limit = self.config.geometry.z_lower_limit


    
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
    

    def loss_measurements(self,x_n,y_n,c_n):
        x= x_n.clone().detach().requires_grad_(True)
        y= y_n.clone().detach().requires_grad_(True)
        c = c_n.clone().detach().requires_grad_(False)
        
        
        u = self.forward(x,y)        

        if self.config.model.measurement_spatial_weight:
            # measurement_weight = 1- torch.square(2* (y-self.z_center)/(self.z_max-self.z_min))

            measurement_weight = (y-self.z_min)/(self.z_max-self.z_min)
            mask1 = measurement_weight>self.z_upper_limit
            mask2 = measurement_weight<self.z_lower_limit
            measurement_weight = torch.ones_like(measurement_weight)
            measurement_weight[mask1]=0
            measurement_weight[mask2]=0
            # mask= mask1 & mask2
            # measurement_weight[~mask]=0

            losses = self.loss_function_no_reduction(u, c)
            weighted_loss = torch.mean(measurement_weight * losses)

        else:
            u = self.forward(x,y)
            weighted_loss = self.loss_function(u, c)    
        return weighted_loss 
    
    def loss_drichlet(self,x_b,y_b,c):
        x = x_b.clone().detach().requires_grad_(True)
        y = y_b.clone().detach().requires_grad_(True)

        # mask=y_b>0.005
        # c[mask]=1
        # c[~mask]=0
        loss_u = self.loss_function(self(x,y), c)
                
        return loss_u #* self.config.model.loss_weight.loss_bc
    

    def loss_newmann(self,x_n,y_n):
        x= x_n.clone().detach().requires_grad_(True)
        y= y_n.clone().detach().requires_grad_(True)
        
        u = self.forward(x,y)

        flux=self._flux(x,y,u)
        loss_newmann = self.loss_function(flux, torch.zeros_like(flux).to(device))

        return loss_newmann 