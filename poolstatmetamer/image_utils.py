#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:07:46 2019

A collection of utility routines for loading, plotting, and saving images stored as torch.Tensors

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import math
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors
import torch
import torchvision.transforms as transforms
import torchvision.utils
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

_default_figsize = [9,6]
if torch.__version__ <= '1.7':
    plt.rcParams["mpl_toolkits.legacy_colorbar"] = False  # This removes a deprecation warning in matplotlib 3.2 (will probably become unneeded in some future release)

# Load an image, convert it to grayscale, and return it as a tensor
def load_image_gray(filename):
#    print(f'cwd:{os.getcwd()} filename:{filename}')
    pil_image = Image.open(filename).convert('L')
    tensor_image = transforms.ToTensor()(pil_image)
    if tensor_image.dim()==3: tensor_image = tensor_image.unsqueeze(0)  # Make sure result has pytorch standard 4 dimensions
    return tensor_image

# Load an image and return it as a tensor
def load_image_rgb(filename):
    pil_image = Image.open(filename).convert('RGB')
    tensor_image = transforms.ToTensor()(pil_image)
    if tensor_image.dim()==3: tensor_image = tensor_image.unsqueeze(0)  # Make sure result has pytorch standard 4 dimensions
    return tensor_image

# Perform a simplisitic gamma correction on image taking it from linear color space to typical stored image space (eg .png or .jpg)
def gamma_correct_image(image,gamma=2):
    return torch.pow(image,1/gamma)    

# Save an tensor image to file
def save_image(tensor,filename,gamma=None,verbose=True):
    if gamma is not None:
        tensor = gamma_correct_image(tensor,gamma)         #Perform gamma correction before saving (recommended if image was in a linear color space)
    elif not tensor.is_cuda:
        tensor = tensor.clone()   #Save image seems to sometimes modify the image?!? (*255)
    if tensor.is_cuda: tensor = tensor.cpu()   # Make sure tensor is copied to cpu first if needed
    torchvision.utils.save_image(tensor, filename)  #WARNING!!: modifies tensor in some cases
    if verbose: print(f"Saved image to {filename}")

# Display an image onscreen
def show_image(tensor):
    pil = transforms.ToPILImage()(tensor.squeeze())
    pil.show()
    
    
# An iterable set of frames defined by a template string and a range    
# For example LoadFrames("frame{i:03d}.png",range(0,30)) iterates through 30 sequential frames ("frame000.png" ... "frame029.png")
class LoadFrames():
    def __init__(self,source_template,source_range):
        self.source_template = source_template
        self.source_range = source_range
    def __iter__(self):
        for (framenum,sourcenum) in enumerate(self.source_range):
            source = self.source_template.format(i=sourcenum)
            yield load_image_rgb(source)
        
class LoadFramesGray():
    def __init__(self,source_template,source_range):
        self.source_template = source_template
        self.source_range = source_range
    def __iter__(self):
        for (framenum,sourcenum) in enumerate(self.source_range):
            source = self.source_template.format(i=sourcenum)
            yield load_image_gray(source)
        
# A generator that will load frames from a movie file and return them one at a time as RGB tensor images
class LoadMovie():
    def __init__(self,filename,framerange=None):
        self.filename = filename
        self.framerange = framerange
    def __iter__(self):
        print(f'Loading movie from {self.filename}')
        for num,frame in enumerate(imageio.get_reader(self.filename)):
            if (self.framerange is not None) and (not num in self.framerange): continue   # skip any frames not in the given range
            pil_image = Image.fromarray(frame)    # For consistency with single image loading, we use PIL to convert the input to the appropriate color space
            pil_image = pil_image.convert('RGB')
            yield transforms.ToTensor()(pil_image).unsqueeze(0) # convert to pytorch standard 4 dimensional format

# A generator that will load frames from a movie file and return them one at a time as RGB tensor images
class LoadMovieGray():
    def __init__(self,filename,framerange=None):
        self.filename = filename
        self.framerange = framerange
    def __iter__(self):
        print(f'Loading grayscale movie from {self.filename}')
        for num,frame in enumerate(imageio.get_reader(self.filename)):
            if (self.framerange is not None) and (not num in self.framerange): continue   # skip any frames not in the given range
            pil_image = Image.fromarray(frame)    # For consistency with single image loading, we use PIL to convert the input to the appropriate color space
            pil_image = pil_image.convert('L')
            yield transforms.ToTensor()(pil_image).unsqueeze(0) # convert to pytorch standard 4 dimensional format
        
# Makes a purple to green color ramp with black in the middle, good for showing negative&positive parts of an image
def make_purple_green_cmap():
    N = 255  # Value should be odd so center value will be black
    ramp = np.clip(np.linspace(-1,1,N), 0, 1) #List of number half zero, half linear ramp to one
    ramp = np.power(ramp, 1/2)       #approx inverse gamma correct to get more perceptually linear scale
    ramp_rev = ramp[::-1]            #reversed ramp
    vals = np.ones((N, 4))
    vals[:, 0] = ramp_rev
    vals[:, 1] = ramp
    vals[:, 2] = ramp_rev
    newcmap = matplotlib.colors.ListedColormap(vals)
    return newcmap
    
# Plot the image here using matplotlib.
def plot_image(tensor,title=None,center_zero=None,colorbar=True,show=True,savefile=None):
    plt.figure(figsize=_default_figsize)
    if title is not None:
        plt.title(title)     #add plot title if one was provided
    im_aspect = tensor.size(-1)/tensor.size(-2)
    if (tensor.dim()==4): tensor = tensor.squeeze(0)  # pytorch likes to add batch dimension as first of four
    if tensor.dim()==2 or tensor.size()[0]==1:
        if center_zero==None: 
            center_zero = tensor.min().item() < -0.1*tensor.max().item()   # Auto-detect if plotting signed functino
        if (center_zero):  # Plot as diverging range symmetrically about zero (so zero maps to middle of colormap)
            limit = max(tensor.max(),-tensor.min())
            cm = make_purple_green_cmap()
            im = plt.imshow(tensor.detach().squeeze().cpu().numpy(), cmap=cm, vmin=-limit, vmax=limit)
        else:
            #if its grayscale, then squeeze to make it just a 2d array with gray colormap
            im = plt.imshow(tensor.detach().squeeze().cpu().numpy(), cmap='gray')
    else:
        # imshow needs a numpy array with the channel dimension as the the last dimension so we have to transpose things.
        im = plt.imshow(tensor.detach().cpu().numpy().transpose(1, 2, 0))
    cbar_fraction = 0.05 / min(im_aspect,4)
    if colorbar: plt.colorbar(im,fraction=cbar_fraction,pad=0.02)
#    if colorbar: plt.colorbar(fraction=0.05,pad=0.1)
#    if colorbar: plt.colorbar()
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, bbox_inches='tight')     # can be saved to .png or pdf file
        print(f'Image plot {title} written to {savefile}')
    if show:
        plt.show()
    else:
        plt.close()        
    
    
def _subplot_image(subplot,tensor,cmap=None,vmin=None,vmax=None):
    if tensor.dim()==2 or tensor.size()[0]==1:
        im = subplot.imshow(tensor.squeeze().detach().cpu().numpy(), cmap=cmap,vmin=vmin,vmax=vmax)
    else:
        # imshow needs a numpy array with the channel dimension as the the last dimension so we have to transpose things.
        im = subplot.imshow(tensor.detach().cpu().numpy().transpose(1, 2, 0))
    #subplot.set_xticks([])
    #subplot.set_yticks([]) #remove ticks and tick labels
    return im
        
    
# Convert to each channel to separate image, concatenate them, and plot combined grayscale image
def plot_image_channels(image,arrange2d=False,**kwargs):
    if (image.dim()==4):
        # If it is an image stack, plot each image in stack separately
        for img in image.unbind(): plot_image_channels(img,**kwargs)
    else:
        if arrange2d and image.size(0)==4:
            image = torch.cat(image.chunk(2),-2)
        combo = torch.cat(image.unbind(),-1)
        plot_image(combo,**kwargs)
        
# Convert image stack in tensor to list of images, concatenate them into one big image and plot the result
def plot_image_stack(imagestack,**kwargs):
    if imagestack.dim() < 4 or imagestack.size(0)==1: return plot_image(imagestack,**kwargs) #if it is not a stack, try it as a single image
    combo = torch.cat(imagestack.unbind(),-1)
    plot_image(combo,**kwargs)
    
def plot_images_alt(tensortuple,colorbar=True):
    num = len(tensortuple)
    fig, ax = plt.subplots(nrows=1, ncols=num, figsize=[12,5])
    for i,tensor in enumerate(tensortuple):
        _subplot_image(ax[i],tensor)
        ax[i].set_aspect(1,adjustable='box')
    #if colorbar: plt.colorbar()
    plt.tight_layout()
    plt.show()
    #plt.savefig("test.png", bbox_inches='tight')  


# Plot a list of images as a grid of images with a shared colorbar
def plot_images(tensorlist, colorbar=True, center_zero=None, title=None ,show=True,savefile=None,
                num_rows=None):
    if torch.is_tensor(tensorlist):  # Handle case where tensor is passed instead of a list
        if tensorlist.dim() == 4 and tensorlist.size(0) > 1:
            tensorlist = tensorlist.unbind()  # If contained multiple images, unbind them into a list
        else:
            tensorlist = (tensorlist,)  # otherwise convert to singleton list
    if len(tensorlist) == 1:
        if isinstance(tensorlist[0],(list,tuple)) and len(tensorlist[0]) == 1:
            tensorlist = tensorlist[0]  #if its list and sublist are singletons then remove outer list
        # If it is just a single image, use ordinary plot function
        if torch.is_tensor(tensorlist[0]):
            plot_image(tensorlist[0], center_zero=center_zero, title=title, show=show, savefile=savefile)
            return
    # Set up figure and image grid
    ncol = len(tensorlist)
    nrow = 1
    vmin = math.inf
    vmax = -math.inf
    if isinstance(tensorlist[0], (list, tuple)):
        nrow = ncol
        ncol = len(tensorlist[0])
        flatlist = []
        for sublist in tensorlist:
            flatlist += sublist
    else:
        flatlist = tensorlist
        if num_rows is None: num_rows = int(math.sqrt(len(flatlist)))
        if num_rows > 1:
            nrow = num_rows
            ncol = (len(flatlist) + num_rows - 1)//num_rows            
    for im in flatlist:
        vmin = min(im.min().item(), vmin)
        vmax = max(im.max().item(), vmax)
    if center_zero == None:
        center_zero = vmin < -0.1 * vmax  # Auto-detect if plotting signed function
    if center_zero:
        cmap = make_purple_green_cmap()
        vmax = max(vmax, -vmin)
        vmin = -vmax
    else:
        cmap = 'gray'

    fig = plt.figure(figsize=_default_figsize)
    cbar_fraction = 5/min(4,flatlist[0].size(-1)/flatlist[0].size(-2))
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(nrow, ncol),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
#                     cbar_size="5%",
                     cbar_size=f"{cbar_fraction}%",
                     cbar_pad=0.1,
                     )
    # Add data to image grid
    imlist = []
    for i, ax in enumerate(grid):
        im = _subplot_image(ax, flatlist[i], cmap=cmap, vmin=vmin, vmax=vmax)
        imlist.append(im)
    # Colorbar
    ax.cax.colorbar(imlist[0])
    ax.cax.toggle_label(True)
#    plt.tight_layout()    # Works, but may still require rect paramater to keep colorbar labels visible
    if title is not None:
        #        ax.set_title(title)     #add plot title if one was provided
        #x0,y0,x1,y1 = grid[0].get_position()
#        print(imlist[0].get_window_extent())
#        print([method_name for method_name in dir(grid) if callable(getattr(grid, method_name))])
#        y1 = grid[0].get_position().y1
#        plt.suptitle(title,y=y1)
#        plt.suptitle(title)
        if ncol == 1 or True:
            grid[0].set_title(title)
        else:
            #fig.suptitle(title)  #suptitle does not work well with imagegrid, is often oddly spaced and much too high
            grid[0].set_title(title)
            #grid[ncol//2].set_title(title)
            
    #The standard value of 'top' is 0.9, tune a lower value, e.g., 0.8
    #plt.subplots_adjust(top=0.9)     
    if savefile:
        plt.savefig(savefile, bbox_inches='tight')     # can be saved to .png or pdf file
        print(f'Image plot {title} written to {savefile}')
    if show:
        plt.show()
    else:
        plt.close()        

# Plot the image here using matplotlib.
def plot_histogram(tensor,bins=100,title=None,show=True,savefile=None):
    plt.figure(figsize=_default_figsize)
    if title is not None:
        plt.title(title)     #add plot title if one was provided
        
    plt.hist(tensor.flatten().cpu().numpy(),bins=bins)
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, bbox_inches='tight')     # can be saved to .png or pdf file
        print(f'Image plot {title} written to {savefile}')
    if show:
        plt.show()
    else:
        plt.close()        
