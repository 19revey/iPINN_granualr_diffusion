import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import os
from box.exceptions import BoxValueError
from box import ConfigBox
import yaml

from granular.visualizer import Visualizer

import PIL
import imageio

# LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE="lastrun.log"
LOG_FILE_PATH=os.path.join("artifacts","logs",LOG_FILE)


if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)

os.makedirs(os.path.dirname(LOG_FILE_PATH),exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(filename)s:%(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)



def read_yaml(path_to_yaml: str) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

def plot_points(t,z,bc):
    plt.figure(figsize=(6, 4))
    plt.scatter(t, z, s=.2, marker=".", c="r", label="CP")
    for key in bc.keys():
        x, y, t = bc[key]
        plt.scatter(x, y,  marker="x", c="k", label="BDP")

    plt.xlabel("t (s)")
    plt.ylabel("z/h")
    plt.tight_layout()
    os.makedirs("artifacts/figures", exist_ok=True)
    plt.savefig("artifacts/figures/points.png")
    plt.close()


    
def plot_animation_loss(train_times,losses, **kwargs):

    line_color = kwargs.get('line_color', 'gray')
    gray_alpha = kwargs.get('gray_alpha', 0.5)
    lw = kwargs.get('lw', 0.5)
    interval = kwargs.get('interval', 10)
    filename = kwargs.get('filename', 'loss_vs_epoch_highlighted.gif')
    # skip = kwargs.get('skip', 10)
    log = kwargs.get('log', False)
    x_label = kwargs.get('x_label', 'Epoch')
    y_label = kwargs.get('y_label', 'Loss')


    epochs = len(losses)  # Number of epochs
    loss_values = losses

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(3, 2))
    # ax.set_xlim(0, 1)  # Set the x-axis limit for training time
    # ax.set_ylim(0, 1)   # Set the y-axis limit for loss
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Epoch vs. Loss')
    if log:
        ax.set_yscale('log')


    # Plot the entire loss curve in gray
    ax.plot(train_times, loss_values, color=line_color, lw=lw, alpha=gray_alpha)

    # Initialize the line for the animated part and the marker for the current point
    line, = ax.plot([], [], lw=0.5, color='blue')
    marker, = ax.plot([], [], 'ro',markersize=2)  # 'ro' means red color with circle marker

    # Initialize the line
    def init():
        line.set_data([], [])
        marker.set_data([], [])
        return line, marker

    # Update function for animation
    def update(frame):
        # Update the animated line with solid blue line for previous points
        line.set_data(train_times[:frame], loss_values[:frame])
        # Update the current point marker
        marker.set_data(train_times[frame-1], loss_values[frame-1])  # Mark the last point in the current frame
        return line, marker

    # Create the animation
    frame_indices = np.arange(0, epochs)  # Update every 10 frames
    ani = FuncAnimation(fig, update, frames=frame_indices, init_func=init, blit=True, repeat=False, interval=interval)

    # To save the animation as a video or gif file
    ani.save(filename, writer='pillow')



def save_animation(epoches, x,y,results,losses, **kwargs):

    fps = kwargs.get('fps', 30)

    # if os.path.exists('artifacts/temp'):
    #     os.removedirs('artifacts/temp')

    os.makedirs(os.path.dirname('artifacts/temp/'),exist_ok=True)
    


    for i,epoch in enumerate(epoches):
        plt.figure(figsize=(6, 5))
        
  
        plt.subplot(222)
        Visualizer.plot_individual_contour(x,y, results['prediction'][i],title="PINN")    
        plt.subplot(224)
        Visualizer.plot_individual_contour(x,y, results['residue'][i],title="PDE Residue")

        plt.subplot(221)
        plot_individual_loss(epoches,losses['total_loss'],current_epoch=i,x_label="Epoch",y_label="Loss",title="Epoch vs. Loss")
        plt.subplot(223)
        plot_individual_loss(losses['lambd'],losses['total_loss'],current_epoch=i,x_label="C_d",y_label="Loss",title="C_d vs. Loss")

        plt.tight_layout()
        plt.savefig(f"artifacts/temp/{i:03d}.png")
        plt.close()
        
    frames=[] 
    for i,epoch in enumerate(epoches):
        frame=PIL.Image.open(f"artifacts/temp/{i:03d}.png")  
        frame=np.asarray(frame)
        frames.append(frame) 
    imageio.mimsave('artifacts/animations/combined.gif', frames, fps=fps, loop=0) 
    # imageio.mimsave('artifacts/animations/combined.mp4', frames, fps=fps, codec='libx264')


def plot_animation_contour(x, y,c, **kwargs):
    Visualizer.plot_individual_contour(x,y,c)

def plot_individual_loss(epoches, losses, **kwargs):
    log = kwargs.get('log', True)
    x_label = kwargs.get('x_label', 'Epoch')
    y_label = kwargs.get('y_label', 'Loss')
    title = kwargs.get('title', 'Epoch vs. Loss')

    current_epoch = kwargs.get('current_epoch', None)
    
    plt.plot(epoches, losses, color='k', lw=0.5)

    if current_epoch:
        plt.plot(epoches[:current_epoch], losses[:current_epoch], color='blue', lw=1)
        plt.plot(epoches[current_epoch-1], losses[current_epoch-1], 'ro', markersize=4)

    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if log:
        plt.gca().set_yscale('log')
    
    plt.plot()