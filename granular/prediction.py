
from torch import nn
from box import ConfigBox
import matplotlib.pyplot as plt
import numpy as np
from box import ConfigBox
import torch
import os
from granular.points import Points

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(PINN, config:ConfigBox): 

    t_min = config.geometry.t_min
    t_max = config.geometry.t_max
    z_min = config.geometry.z_min
    z_max = config.geometry.z_max

    ### plotting
    
    #
    nx=50
    ny=500
    X = np.linspace(t_min, t_max, nx)
    Y = np.linspace(z_min,z_max, ny)
    X0, Y0 = np.meshgrid(X, Y)

    X = X0.reshape([ny*nx, 1])
    Y = Y0.reshape([ny*nx, 1])

    X_T = torch.from_numpy(X).float().to(device)
    Y_T = torch.from_numpy(Y).float().to(device)
    # X_F = torch.cat((X_T,Y_T),dim=1).float().to(device)

    S = PINN(X_T,Y_T)
    S = S.cpu().detach().numpy().reshape(ny, nx)

    pde_residue = PINN.get_PDE_residue(X_T,Y_T)
    pde_residue = pde_residue.cpu().detach().numpy().reshape(ny, nx)

    
    ## DEM
    if config.model.inverse:
        points = Points(config)
        x_m,y_m,c_m=points.get_matlab_data()
        x_m1,y_m1,c_m1=points.load_matlab_file()
        measurement_residue = PINN.get_measurement_residue(x_m1, y_m1, c_m1)
        measurement_residue = measurement_residue.cpu().detach().numpy().reshape(c_m.shape[0], c_m.shape[1])


    if not config.model.inverse:

        plt.figure(figsize=(5, 4))
        plt.subplot(111)
        plt.pcolormesh(X0, Y0, 1.*S,vmin=0,vmax=1,  cmap="magma")
        plt.colorbar()
        plt.xlabel("t")
        plt.ylabel("z")
        plt.title("PINN")
        plt.tight_layout()
        # plt.axis("square")
    else:
        plt.figure(figsize=(12, 8))

        plt.subplot(222)
        plt.pcolormesh(X0, Y0, 1.*S,vmin=0,vmax=1,  cmap="magma")
        plt.colorbar()
        plt.xlabel("t")
        plt.ylabel("z")
        plt.title("PINN")
        plt.tight_layout()

        plt.subplot(221)
        plt.pcolormesh(x_m, y_m, c_m,vmin=0,vmax=1,  cmap="magma")
        plt.colorbar()
        plt.xlabel("t")
        plt.ylabel("z")
        plt.title("DEM")
        plt.tight_layout()
        # plt.axis("square")

        plt.subplot(223)
        plt.pcolormesh(X0, Y0, pde_residue, vmin=0, vmax=0.05, cmap="magma")
        plt.colorbar()
        plt.xlabel("t")
        plt.ylabel("z")
        plt.title("PDE residue")
        plt.tight_layout()
        # plt.axis("square")
        #     
        plt.subplot(224)
        
        plt.pcolormesh(x_m, y_m,  measurement_residue, vmin=0, vmax=0.1, cmap="magma")
        plt.colorbar()
        plt.xlabel("t")
        plt.ylabel("z")
        plt.title("Error")
        plt.tight_layout()
        # plt.axis("square")   

    os.makedirs("artifacts/figures", exist_ok=True)
    plt.savefig("artifacts/figures/pred_contour.png")
    plt.close()


    S_ = S.reshape([ny, nx])


    height = 3
    frames_val = np.array([0,0.25,  0.5, +.75, +1])
    frames = [*map(int, frames_val * (nx-1))]

    if config.model.inverse:
        frames1 = [*map(int, frames_val * (c_m.shape[1]-1))]
    

    
    plt.figure("", figsize=(len(frames)*height, 1*height))

    plt.clf()
    for i, var_index in enumerate(frames):
        plt.subplot(1, len(frames), i+1)
        plt.title(f"t = {frames_val[i]:.2f}")
        plt.plot( S_[:,var_index],Y0[:,var_index], "r--", lw=4., label="pinn")
        if config.model.inverse:
            plt.plot( c_m[:,frames1[i]],y_m[:], "b", lw=2., label="DEM")
        # plt.plot( z_simc[:,frames1[i]],Y_simc[:, frames1[i]], "b", lw=2., label="DEM")
        plt.ylim(z_min, z_max)
        plt.xlim(0, 1)
        plt.xlabel("c")
        plt.ylabel("h")
        plt.tight_layout()
        plt.legend()

    plt.savefig("artifacts/figures/pred_profiles.png")
    plt.close()

    # print(S_[:,-1].shape)


