
from box import ConfigBox
from scipy.stats import qmc
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


class Points:
    def __init__(self, config:ConfigBox):
        self.t_min = config.geometry.t_min
        self.t_max = config.geometry.t_max
        self.z_min = config.geometry.z_min
        self.z_max = config.geometry.z_max
        self.n_data_per_bc = config.geometry.n_data_per_bc
        self.n_data_collocation = config.geometry.n_data_collocation

        self.cl = config.particle.cl
        
        self.file_path = config.model.mat_file

    def boundary_points(self):

        n_bc = 4
        n_data_per_bc = self.n_data_per_bc
        #
        engine = qmc.LatinHypercube(d=1)
        data = np.zeros([4, n_data_per_bc, 3])

        for i, j in zip(range(n_bc), [self.t_min, self.t_max, self.z_min,self.z_max]):
            points_z = engine.random(n=n_data_per_bc)[:, 0]*(self.z_max-self.z_min) + self.z_min
            points_t = engine.random(n=n_data_per_bc)[:, 0]*(self.t_max-self.t_min) + self.t_min
            if i < 2:
                data[i, :, 0] = j
                data[i, :, 1] = points_z
            else:
                data[i, :, 0] = points_t
                data[i, :, 1] = j
        
        left=data[0, :, :]
        right=data[1, :, :]
        bottom=data[2, :, :]
        top=data[3, :, :]

        left[:,-1]=self.cl

        bc = {}

        x_d_left, y_d_left, t_d_left = map(lambda x: np.expand_dims(x, axis=1),[left[:, 0], left[:, 1], left[:, 2]])
        x_d_left, y_d_left, t_d_left = map(lambda x: torch.from_numpy(x), [x_d_left, y_d_left, t_d_left])
        bc['left']=(x_d_left, y_d_left, t_d_left)

        right[:,2]=0
        x_d_right, y_d_right, t_d_right = map(lambda x: np.expand_dims(x, axis=1), [right[:, 0], right[:, 1], right[:, 2]])
        x_d_right, y_d_right, t_d_right = map(lambda x: torch.from_numpy(x), [x_d_right, y_d_right, t_d_right])
        bc['right']=(x_d_right, y_d_right, t_d_right)

        bottom[:,2]=0
        x_d_bottom, y_d_bottom, t_d_bottom = map(lambda x: np.expand_dims(x, axis=1), [bottom[:, 0], bottom[:, 1], bottom[:, 2]])
        x_d_bottom, y_d_bottom, t_d_bottom = map(lambda x: torch.from_numpy(x), [x_d_bottom, y_d_bottom, t_d_bottom])
        bc['bottom']=(x_d_bottom, y_d_bottom, t_d_bottom)

        top[:,2]=0
        x_d_top, y_d_top, t_d_top = map(lambda x: np.expand_dims(x, axis=1), [top[:, 0], top[:, 1], top[:, 2]])
        x_d_top, y_d_top, t_d_top = map(lambda x: torch.from_numpy(x), [x_d_top, y_d_top, t_d_top])
        bc['top']=(x_d_top, y_d_top, t_d_top)

        return bc

    def collocation_points(self):
        Nc = self.n_data_collocation
        engine = qmc.LatinHypercube(d=2)
        colloc = engine.random(n=Nc)
        bounds = np.array([[self.t_min, self.t_max], [self.z_min, self.z_max]])  # 2D bounds array for t and z
        colloc_scaled = qmc.scale(colloc, bounds[:, 0], bounds[:, 1])
        
        #decompose from (N,2) to (N,1) and (N,1)
        x_c, y_c = map(lambda x: np.expand_dims(x, axis=1), 
                    [colloc_scaled[:, 0], colloc_scaled[:, 1]])
        x_c, y_c = map(lambda x: torch.from_numpy(x), [x_c, y_c])
        return x_c, y_c
    
    def get_matlab_data(self):
        mat_data = scipy.io.loadmat(self.file_path)
        t_len,x_len=mat_data['data'].shape
        x_sim= np.linspace(self.t_min, self.t_max, t_len)
        y_sim= np.linspace(self.z_min, self.z_max, x_len)
        X_sim, Y_sim = np.meshgrid(x_sim, y_sim)
        z_sim=mat_data['data'].transpose()
        z_sim=np.nan_to_num(z_sim, nan=0)
        return x_sim, y_sim, z_sim
    
    def load_matlab_file(self):
        # mat_data = scipy.io.loadmat(self.file_path)
        # t_len,x_len=mat_data['data'].shape
        # x_sim= np.linspace(self.t_min, self.t_max, t_len)
        # y_sim= np.linspace(self.z_min, self.z_max, x_len)
        x_sim, y_sim, z_sim = self.get_matlab_data()
        X_sim, Y_sim = np.meshgrid(x_sim, y_sim)
        # z_sim=mat_data['data'].transpose()
        # z_sim=np.nan_to_num(z_sim, nan=0)
        # print(z_sim.shape)
        # plt.contourf(X_sim, Y_sim, z_sim,levels=100, cmap="magma")
        # plt.colorbar()
        # plt.savefig('artifacts/figures/DEM.png')
        # plt.close()


        X_sim = X_sim.reshape(-1,1)
        Y_sim = Y_sim.reshape(-1,1)
        z_sim = z_sim.reshape(-1,1)

    
        x_d_top, y_d_top, t_d_top = map(lambda x: torch.from_numpy(x), [X_sim, Y_sim, z_sim])
        return x_d_top, y_d_top, t_d_top