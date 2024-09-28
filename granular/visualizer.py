import numpy as np
from box import ConfigBox
from granular.points import Points
import torch
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Visualizer:
    def __init__(self, pinn,losses, config: ConfigBox):
        self.pinn = pinn
        self.config = config
        self.losses = losses
        
    def update(self,pinn):
        self.pinn = pinn

    def _get_mat_data(self):
        points = Points(self.config)
        x_m,y_m,c_m=points.get_matlab_data()
        x_m1,y_m1,c_m1=points.load_matlab_file()
        measurement_residue = self.pinn.get_measurement_residue(x_m1, y_m1, c_m1)
        measurement_residue = measurement_residue.cpu().detach().numpy().reshape(c_m.shape[0], c_m.shape[1])
        return x_m,y_m,c_m,measurement_residue

    def _get_coll_data(self,nx=50,ny=500):

        self.t_min = self.config.geometry.t_min
        self.t_max = self.config.geometry.t_max
        self.z_min = self.config.geometry.z_min
        self.z_max = self.config.geometry.z_max

        X = np.linspace(self.t_min, self.t_max, nx)
        Y = np.linspace(self.z_min,self.z_max, ny)
        X0, Y0 = np.meshgrid(X, Y)

        X = X0.reshape([ny*nx, 1])
        Y = Y0.reshape([ny*nx, 1])

        X_T = torch.from_numpy(X).float().to(device)
        Y_T = torch.from_numpy(Y).float().to(device)

        S = self.pinn(X_T,Y_T)
        S = S.cpu().detach().numpy().reshape(ny, nx)

        pde_residue = self.pinn.get_PDE_residue(X_T,Y_T)
        pde_residue = pde_residue.cpu().detach().numpy().reshape(ny, nx)

        return X0,Y0,S,pde_residue

    def plot_contour(self):
        x_m,y_m,c_m,measurement_residue = self._get_mat_data()
        x_p,y_p,c_p,pde_residue = self._get_coll_data()

        plt.figure(figsize=(12, 12))
        plt.subplot(321)
        Visualizer.plot_individual_contour(x_p,y_p,c_p,title="PINN")
        plt.subplot(322)
        Visualizer.plot_individual_contour(x_m,y_m,c_m,title="DEM")
        plt.subplot(323)
        Visualizer.plot_individual_contour(x_p,y_p,pde_residue,title="PDE Residue")
        plt.subplot(324)
        Visualizer.plot_individual_contour(x_m,y_m,measurement_residue,title="Measurement Residue")
        plt.subplot(325)
        Visualizer.plot_individual_loss(self.losses['epoches'],self.losses['total_loss'],log=True)
        plt.subplot(326)
        Visualizer.plot_individual_loss(self.losses['lambd'],self.losses['total_loss'],log=False)

        os.makedirs("artifacts/figures", exist_ok=True)
        plt.savefig("artifacts/figures/pred_contour.png")
        plt.close()

    def plot_contour_for_animation(self):
        x_m,y_m,c_m,measurement_residue = self._get_mat_data()
        x_p,y_p,c_p,pde_residue = self._get_coll_data()

        # plt.figure(figsize=(12, 12))
        plt.subplot(321)
        Visualizer.plot_individual_contour(x_p,y_p,c_p,title="PINN")
        plt.subplot(322)
        Visualizer.plot_individual_contour(x_m,y_m,c_m,title="DEM")
        plt.subplot(323)
        Visualizer.plot_individual_contour(x_p,y_p,pde_residue,title="PDE Residue")
        plt.subplot(324)
        Visualizer.plot_individual_contour(x_m,y_m,measurement_residue,title="Measurement Residue")
        plt.subplot(325)
        Visualizer.plot_individual_loss(self.losses['epoches'],self.losses['total_loss'],log=True)
        plt.subplot(326)
        Visualizer.plot_individual_loss(self.losses['lambd'],self.losses['total_loss'],log=False)



    @staticmethod
    def plot_individual_contour(X0,Y0,S,title="PINN"):
        if title in ["PDE Residue","Measurement Residue"]:
            plt.pcolormesh(X0, Y0, S, norm=LogNorm(vmin=0.01, vmax=1),cmap="magma")
        else:
            plt.pcolormesh(X0, Y0, 1.*S,vmin=0,vmax=1,  cmap="magma")
        plt.colorbar(fraction=0.04, pad=0.06)
        plt.xlabel("t")
        plt.ylabel("z")
        plt.title(title)
        plt.tight_layout()


    @staticmethod
    def plot_individual_loss(epoches, losses,**kwargs):
        log = kwargs.get('log', False)
        x_label = kwargs.get('x_label', 'Epoch')
        y_label = kwargs.get('y_label', 'Loss')
        title = kwargs.get('title', 'Epoch vs. Loss')
        
        plt.plot(epoches, losses, color='k', lw=0.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if log:
            plt.gca().set_yscale('log')
        plt.tight_layout()