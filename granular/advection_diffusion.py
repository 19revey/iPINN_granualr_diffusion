
import torch.nn as nn
import torch
import math
import numpy as np
from granular.utils import logging
from box import ConfigBox
import torch.optim as optim
from granular.base_model import Base_net, device

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(Base_net):
    def __init__(self, config:ConfigBox):
        super(Net, self).__init__(config)



    def _intruder_force(self):


        _intruder_l=(1-1.43*np.exp(-self.rd/0.92))*(1+3.55*np.exp(-self.rd/2.94))*self.phi
        _intruder_s=(1-1.43*np.exp(-self.rds/0.92))*(1+3.55*np.exp(-self.rds/2.94))*self.phi

        intruder_l=torch.tensor(_intruder_l, dtype=torch.float32, requires_grad=True).to(device) 
        intruder_s=torch.tensor(_intruder_s, dtype=torch.float32, requires_grad=True).to(device) 

        return intruder_l, intruder_s
    

    def _flux(self,x,y,u):

        if self.config.model.inverse:
            intruder_l=self.lambd[0]
            intruder_s=self.lambd[1]
            
        else:
            intruder_l, intruder_s = self._intruder_force()

        depth = self.config.geometry.z_max - y
        p = self.rho*self.phi *self.g* depth   + self.p0
        inert=self.gamma*(u*self.dl+(1-u)*self.ds)/torch.sqrt(p/self.rho)
        mu_eff=0.364+(0.772-0.364)/(0.434/inert+1)
        eta=mu_eff*p/self.gamma

        c_dep=torch.tanh(torch.abs(intruder_s)*(1-u)/(u))
        c_dep= torch.clamp(c_dep, 0, 1)

        mixture_l=(intruder_l)*c_dep
        cd=(2-7*math.exp(-2.6*self.rd))+0.57*inert*self.rd
        wseg=mixture_l*self.ml*self.g / (cd*math.pi*eta*self.dl)
        
        # logging.info(f"mixture_l: {mixture_l[0].item()}, wseg: {wseg[0].item()}, c: {u[0].item()}, eta: {eta[0].item()}, p: {p[0].item()}, depth: {depth[0].item()}, inert: {inert[0].item()}")

        u_y = torch.autograd.grad(u,y, torch.ones_like(u).to(device),retain_graph=True, create_graph=True)[0]  
        

        flux = -self.lambd[2]*self.gamma*((1-u)*self.ds+u*self.dl)*((1-u)*self.ds+u*self.dl)*u_y
        # flux = -self.lambd[2]*self.gamma*((1-u)*self.ds+u*self.dl)*((1-u)*self.ds+u*self.dl)/torch.sqrt(inert)*u_y

        # return (1-u)*u-0.1*u_y
        return flux*self.t_scale
    




    def loss_PDE(self, x_c,y_c):
        

        x= x_c.clone().detach().requires_grad_(True)
        y= y_c.clone().detach().requires_grad_(True)
        
        u = self.forward(x,y)


        u_x = torch.autograd.grad(u,x, torch.ones_like(u).to(device),retain_graph=True, create_graph=True)[0]                         

        flux=self._flux(x,y,u)


        flux_y= torch.autograd.grad(flux,y, torch.ones_like(flux).to(device), create_graph=True)[0]
                        
        f =  u_x + flux_y
        
        loss_f = self.loss_function(f,torch.zeros(f.shape).to(device))
                
        return loss_f
    
    def loss_sailency(self, x,y):
        


        u = self.forward(x,y)


        u_x = torch.autograd.grad(u,x, torch.ones_like(u).to(device),retain_graph=True, create_graph=True)[0]                         

        flux=self._flux(x,y,u)


        flux_y= torch.autograd.grad(flux,y, torch.ones_like(flux).to(device), create_graph=True)[0]
                        
        f =  u_x + flux_y
        
        loss_f = self.loss_function(f,torch.zeros(f.shape).to(device))
                
        return loss_f

    def get_PDE_residue(self, x_c,y_c):
        

        x= x_c.clone().detach().requires_grad_(True)
        y= y_c.clone().detach().requires_grad_(True)
        
        u = self.forward(x,y)


        u_x = torch.autograd.grad(u,x, torch.ones_like(u).to(device),retain_graph=True, create_graph=True)[0]                         

        flux=self._flux(x,y,u)


        flux_y= torch.autograd.grad(flux,y, torch.ones_like(flux).to(device), create_graph=True)[0]
                        
        f =  u_x + flux_y
        
                
        return f
    
        

    def get_measurement_residue(self,x_n,y_n,c_n):
        x= x_n.clone().detach().requires_grad_(True).float().to(device)
        y= y_n.clone().detach().requires_grad_(True).float().to(device)
        c = c_n.clone().detach().requires_grad_(False).float().to(device)
        
        
        u = self.forward(x,y)


        return torch.abs(u-c) 
    

    def get_prediction_residue_history(self,x_n,y_n,c_n):
        x= x_n.clone().detach().requires_grad_(True).float().to(device)
        y= y_n.clone().detach().requires_grad_(True).float().to(device)
        c = c_n.clone().detach().requires_grad_(False).float().to(device)
        u = self.forward(x,y)

        return u, torch.abs(u-c) 