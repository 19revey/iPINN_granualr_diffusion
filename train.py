from granular.advection_diffusion import Net
from granular.points import Points
from granular.utils import  plot_points, save_animation
from box import ConfigBox
import torch
import matplotlib.pyplot as plt
import os
from granular.visualizer import Visualizer

from collections import defaultdict
import numpy as np

import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Pinn:
    def __init__(self, config: ConfigBox):
        self.points = Points(config)
        self.model = Net(config).to(device)
        self.config = config
        self.loss_history = defaultdict(list)
        self.prediction_history = defaultdict(list)
        
        self.best_model={'loss':1000.0, 'c_d':1.0, 'model':None}
        # self.best_model['loss']=10000

        # print(self.model._intruder_force())

    def train(self):

        config = self.config
        skip = config.animation.skip


        if config.model.inverse:
            self.file_path = config.model.mat_file
            x_m,y_m,c_m=self.points.load_matlab_file()
            x_i,y_i,c_i = self.points.get_matlab_data()

            x_cpu,y_cpu,c_cpu = self.points.get_matlab_data()

            len_x,len_y=c_cpu.shape

            x_i = y_i.reshape(-1,1)*0 + config.geometry.t_min
            y_i = y_i.reshape(-1,1)  
            c_i = c_i[:,0].reshape(-1,1)
            x_i = torch.from_numpy(x_i).float().to(device)
            y_i = torch.from_numpy(y_i).float().to(device)
            c_i = torch.from_numpy(c_i).float().to(device)

            

        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.model.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', 
            factor=config.model.lr_scheduler_factor, 
            patience=config.model.lr_scheduler_patience
            )

        bc = self.points.boundary_points()
        x_c, y_c = self.points.collocation_points()

        plot_points(x_c, y_c,bc)


        lambd_m = config.model.loss_weight.loss_measurements
        lambd_drich = config.model.loss_weight.loss_drich
        lambd_newmann = config.model.loss_weight.loss_newmann


        for epoch in tqdm(range(config.model.epochs),desc='Training epoch'):
            loss = 0

            # inverse loss
            if config.model.inverse:
                x_m, y_m, c_m = x_m.float().to(device), y_m.float().to(device), c_m.float().to(device)
                loss_measurements = self.model.loss_measurements(x_m, y_m, c_m)
                lambd_m_hat = self.model.compute_gradient_norm(loss_measurements)
                # loss += loss_measurements
                

            # boundtion loss
            for key in bc.keys():
                x_d, y_d, t_d = bc[key]
                x_d, y_d, t_d = x_d.float().to(device), y_d.float().to(device), t_d.float().to(device)
                if key=="left":
                    loss_drichlet = self.model.loss_drichlet(x_d, y_d, t_d)
                    # loss_drichlet = self.model.loss_drichlet(x_i, y_i, c_i)
                    # loss += loss_drichlet
                    lambd_drich_hat = self.model.compute_gradient_norm(loss_drichlet)
                elif key == "top" or key == "bottom":
                    loss_newmann = self.model.loss_newmann(x_d, y_d)
                    # loss += loss_newmann
                    lambd_newmann_hat = self.model.compute_gradient_norm(loss_newmann)
                else:
                    pass
                
            # pde loss
            x_c, y_c = x_c.float().to(device), y_c.float().to(device)
            loss_pde = self.model.loss_PDE(x_c,y_c)


            if config.model.balanced_loss_weight:

                lambd_m = 0.1*lambd_m + 0.9*lambd_m_hat
                lambd_drich = 0.1*lambd_drich + 0.9*lambd_drich_hat
                lambd_newmann = 0.1*lambd_newmann + 0.9*lambd_newmann_hat


            

            if not config.model.include_drich:
                loss_drichlet = loss_drichlet*0
            
            loss = lambd_m*loss_measurements + lambd_drich*loss_drichlet + lambd_newmann*loss_newmann + loss_pde
            # self.loss_history['total_loss'].append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # for name, param in pinn.model.named_parameters():
            #     if 'lambd' in name:
            #         # print(f"weight: {param.data}")
            #         self.loss_history['lambd'].append(param.data[-1].item())

            if epoch % skip == 0:
                tloss=loss.item()

                if config.animation.save_contour_animation:
                    pred, resi = self.model.get_prediction_residue_history(x_m, y_m,c_m)
                    self.prediction_history['prediction'].append(pred.cpu().detach().numpy().reshape(len_x, len_y))
                    self.prediction_history['residue'].append(resi.cpu().detach().numpy().reshape(len_x, len_y))

                for name, param in pinn.model.named_parameters():
                    if 'lambd' in name:
                        self.loss_history['lambd'].append(param.data[-1].item())
                        self.loss_history['total_loss'].append(loss.item())
                        self.loss_history['loss_measurements'].append(loss_measurements.item())
                        self.loss_history['loss_drichlet'].append(loss_drichlet.item())
                        self.loss_history['loss_newmann'].append(loss_newmann.item())
                        self.loss_history['loss_pde'].append(loss_pde.item())
                        self.loss_history['epoches'].append(epoch)

                        if loss.item()< self.best_model['loss']:
                            self.best_model['loss']= loss.item()
                            self.best_model['c_d']=param.data[-1].item()
                            self.best_model['model']=copy.deepcopy(self.model)

                        tqdm.write(f"target parameter: {param.data.reshape(-1)}")



                if config.model.inverse:
                    tqdm.write(f"Learning rate:{optimizer.param_groups[0]['lr']}, Epoch: {epoch}, Loss: {tloss}, measurements: {lambd_m*loss_measurements.item()/tloss:.4f},drichlet: {loss_drichlet.item()/tloss:.4f}, newmann: {loss_newmann.item()/tloss:.4f}, pde: {loss_pde.item()/tloss:.4f}")
                else:
                    tqdm.write(f"Epoch: {epoch}, Loss: {tloss}, drichlet: {lambd_drich*loss_drichlet.item():.4f}, newmann: {lambd_newmann*loss_newmann.item():.4f}, pde: {loss_pde.item():.4f}")
        
        if config.animation.save_contour_animation:
            save_animation(self.loss_history['epoches'],x_cpu,y_cpu,self.prediction_history,self.loss_history,**config.animation)

        return self.best_model['model'], self.loss_history
    
    def save(self):
        torch.save(self.best_model['model'], "artifacts/trained_model/model.pth")
    def load(self):
        self.model = torch.load("artifacts/trained_model/model.pth")
    





if __name__ == "__main__":
    from granular.utils import read_yaml, plot_animation_loss
    from granular.prediction import predict
    params=read_yaml('config.yaml')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help='Path to the input file')
    args = parser.parse_args()



    pinn = Pinn(params)

    if args.train == "new":
        pinn.train()
        pinn.save()
    elif args.train=="continue":
        pinn.load()
        pinn.train()
        pinn.save()
    elif args.train=="load":
        pinn.load()
    else:
        raise ValueError("Invalid argument for --train")
    # pinn.train()
        # pinn.save()

    predict(pinn.model,params)
    
    epoch = pinn.loss_history['epoches']
    if params.animation.save_loss_animation:
        plot_animation_loss(epoch,pinn.loss_history['total_loss'],**params.animation, filename="artifacts/animations/loss_vs_epoch.gif")
        plot_animation_loss(pinn.loss_history['lambd'],pinn.loss_history['total_loss'],x_label="C_d",**params.animation, filename="artifacts/animations/loss_vs_cd.gif")
    
    
    plt.figure(figsize=(6, 4))
    plt.semilogy(pinn.loss_history['total_loss'])
    plt.savefig("artifacts/figures/loss.png")
    plt.close()

    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.semilogy(pinn.loss_history['lambd'], pinn.loss_history['total_loss'],'k.')
    plt.xlabel("C_d")
    plt.ylabel("Error")
    plt.title("L")
    plt.subplot(132)
    plt.semilogy(pinn.loss_history['lambd'], pinn.loss_history['loss_pde'],'r.')
    plt.xlabel("C_d")
    plt.ylabel("Error")
    plt.title("L_PDE")
    plt.subplot(133)
    plt.semilogy(pinn.loss_history['lambd'], pinn.loss_history['loss_measurements'],'b.')
    plt.xlabel("C_d")
    plt.ylabel("Error")
    plt.title("L_data")
    plt.savefig("artifacts/figures/lambd_loss.png")
    plt.close()

    last_1000 = pinn.loss_history['lambd'][-2000:]
    cd= pinn.best_model['c_d'] 
    print(f'best parameter: {cd}')
    print(f'mean lambd: {np.mean(last_1000)}')
    print(f'std lambd: {np.std(last_1000)}')

    # vis = Visualizer(pinn.model,pinn.loss_history,params)
    # vis.plot_contour()

    
    




