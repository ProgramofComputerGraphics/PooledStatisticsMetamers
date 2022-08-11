#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:47:35 2019

Code to implement a variety of simple image filters such gaussian, box, epanechnikov, trapezoid, trigezoid
Generally these are returned as tensors which just big enough to contain them (ie cover their support)
Also some utilities for using such filters as fixed-function convolutional layers in a CNN

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import math
import torch
from imageblends import distance_image
from image_utils import plot_image

def Box2d(width,*,circular=False,normalize_area=False,shrink_fraction=1):
    if circular:
        dist2 = distance_image(width,return_squared_distance=True)
        radiusSquared = width*width*shrink_fraction*shrink_fraction/4
        K = (dist2 <= radiusSquared).float()
    else:  # defalut is to return a separable kernel (in x and y)
        if shrink_fraction == 1:
            # Usually a box filter with just be filled with constant value
            K = torch.ones(width,width)
        else:
            # but if full_fraction < 1 then we create a smaller box filter within the support of the larger filter
            # Usually one would just create a smaller kernel, but in special cases such as some visualization, it can be useful to have a smaller kernel while keeping the same tensor size
            endval = (width - 1)/2
            # create 1d list of distances to the filter center for each pixel center
            x = torch.linspace(-endval,endval,steps=width)
            kern1d = (x.abs() < shrink_fraction*width/2).float()
            # then use outer product to create separable filter
            K = torch.ger(kern1d,kern1d)   # 2d Kernel is outer product of 1d kernels
    if normalize_area:
        K = K / (K.sum())  # Normalize kernel (entries sum to one)
    return K

def _shrink_border(width,shrink_fraction) :
    border = round(0.5*(width - width*shrink_fraction)) # find the width of the border around the shrunken kernel (rounded to nearest integer)
    return border

# Trapezoid shaped kernel (eg, constant for a middle segments with linear ramps at first and last segments)
# full_fraction specifies what fraction of the interval the filter is at full value (eg equal to one)
def Trapezoid2d(width,*,full_fraction=0.5,circular=False,normalize_area=False,shrink_fraction=1):
    shr_border = _shrink_border(width, shrink_fraction)  
    width = width - 2*shr_border  # if using shrink feature, create a smaller kernel and then pad it out to the full size
    radius = width/2
    transition_width = width*(1-full_fraction)/2   # size of a single transition segment (region where filter ramps from zero to one)
    if circular:
        dist = distance_image(width)
        # create circular tent filter with the right slope and then clamp to create flat full value region
        K = ((radius - dist)/transition_width).clamp(min=0,max=1)
    else:  # defalut is to return a separable kernel (in x and y)
        endval = (width - 1)/2
        # create 1d list of distances to the filter center for each pixel center
        x = torch.linspace(-endval,endval,steps=width)
        # create a tent filter with the right slopes
        kern1d = torch.min(radius-x,x+radius)/transition_width
        # then clamp to create the flat full value (value==1) region
        kern1d = kern1d.clamp(max=1)
        # then use outer product to create separable filter
        K = torch.ger(kern1d,kern1d)   # 2d Kernel is outer product of 1d kernels
    if normalize_area:
        K = K / (K.sum())  # Normalize kernel (entries sum to one)
    if shr_border > 0:
        K = torch.nn.functional.pad(K, pad=(shr_border)*4) # pad kernel with zeros around the boundary
    return K    

# Similar to trapezoid filter except uses trigonometric blending function that is C1 continuous
# In 1d these are also sometimes known as Tukey filters
def Trigezoid2d(width,*,full_fraction=0.5,circular=False,normalize_area=False,shrink_fraction=1):
    shr_border = _shrink_border(width, shrink_fraction)  
    width = width - 2*shr_border  # if using shrink feature, create a smaller kernel and then pad it out to the full size
    # First we will create a trapezoid kernel and then apply cosine^2 function to it to get a trigezoid
    radius = width/2
    transition_width = float(width*(1-full_fraction)/2)   # size of a single transition segment (region where filter ramps from zero to one)
    if circular:
        dist = distance_image(width)
        # create circular tent filter with the right slope and then clamp to create flat full value region
        trapK = ((radius - dist)/transition_width).clamp(min=0,max=1)
        K = torch.cos((math.pi/2)*(1-trapK))**2
    else:  # defalut is to return a separable kernel (in x and y)
        endval = (width - 1)/2
        # create 1d list of distances to the filter center for each pixel center
        x = torch.linspace(-endval,endval,steps=width)
        # create a tent filter with the right slopes
        trap_kern1d = torch.min(radius-x,x+radius)/transition_width
        # then clamp to create the flat full value (value==1) region
        trap_kern1d = trap_kern1d.clamp(max=1)
        # then apply cosine^2 function to trapezoid kernel to get trigezoid
        kern1d = torch.cos((math.pi/2)*(1-trap_kern1d))**2
        # then use outer product to create separable filter
        K = torch.ger(kern1d,kern1d)   # 2d Kernel is outer product of 1d kernels
    if normalize_area:
        K = K / (K.sum())  # Normalize kernel (entries sum to one)
    if shr_border > 0:
        K = torch.nn.functional.pad(K, pad=(shr_border)*4) # pad kernel with zeros around the boundary
    return K    

# Similar to trapezoid filter except quadratic blending function that is c1 continuous inside and c0 continuous at the boundary
# If full_fraction==0 then this is equivalent to the epanechnikov kernel
def Epanezoid2d(width,*,full_fraction=0.5,circular=False,normalize_area=False,shrink_fraction=1):
    shr_border = _shrink_border(width, shrink_fraction)  
    width = width - 2*shr_border  # if using shrink feature, create a smaller kernel and then pad it out to the full size
    # First we will create a trapezoid kernel and then apply quadratic function to it to get a Epanezoid
    radius = width/2
    transition_width = float(width*(1-full_fraction)/2)   # size of a single transition segment (region where filter ramps from zero to one)
    if circular:
        dist = distance_image(width)
        # create circular tent filter with the right slope and then clamp to create flat full value region
        trapK = ((radius - dist)/transition_width).clamp(min=0,max=1)
        K = trapK*(2-trapK)
    else:  # defalut is to return a separable kernel (in x and y)
        endval = (width - 1)/2
        # create 1d list of distances to the filter center for each pixel center
        x = torch.linspace(-endval,endval,steps=width)
        # create a tent filter with the right slopes
        trap_kern1d = torch.min(radius-x,x+radius)/transition_width
        # then clamp to create the flat full value (value==1) region
        trap_kern1d = trap_kern1d.clamp(max=1)
        # then apply quadratic function to trapezoid kernel to get trigezoid
        kern1d = trap_kern1d*(2-trap_kern1d)
        # then use outer product to create separable filter
        K = torch.ger(kern1d,kern1d)   # 2d Kernel is outer product of 1d kernels
    if normalize_area:
        K = K / (K.sum())  # Normalize kernel (entries sum to one)
    if shr_border > 0:
        K = torch.nn.functional.pad(K, pad=(shr_border)*4) # pad kernel with zeros around the boundary
    return K    

    # first create a trapezoid filter which we use to set the angles for cosine^2 function in the trigezoid
    trapK = Trapezoid2d(width,full_fraction=full_fraction,circular=circular,normalize_area=False)
    # now compute the cosine^2 function (note: where trapK==1 then K==1)
    K = trapK*(2-trapK)
    if normalize_area:
        K = K / (K.sum())  # Normalize kernel (entries sum to one)
    return K    

# Generate and return a 2D windowed gaussian kernel as a pytorch tensor
def windowed_gaussian(sigma,kernel_radius=None,renormalize=True):
    # By default kernel radius will be three sigma
    if kernel_radius is None:
        kernel_radius = math.ceil(3*sigma)
    
    # Make 1d tensor of integer coordinates: [-kernel_radius,kernel_radius]
    coords1d = torch.arange(start=-kernel_radius,end=kernel_radius+1,dtype=torch.float)
    # Evaluate 1d Gaussian 
    gauss1d = torch.exp(-coords1d*coords1d/(2*sigma*sigma))
    # Compute outer product to get 2D gaussian
    gauss2d = torch.ger(gauss1d,gauss1d)
    if renormalize:
        gauss2d /= gauss2d.sum()           #normalize kernel so it sums to one
    else:
        gauss2d /= 2*math.pi*sigma*sigma   #use default gaussian formula normalization
    return gauss2d


# A fixed 2d convolutional (aka correlational) layer with  a fixed (non-learnable) kernel
class FixedConv2dLayer(torch.nn.Module):
    
    def __init__(self,kernel):
        super().__init__()
        if kernel.size(-1) != kernel.size(-2):
            raise ValueError('non square kernels not yet supported')
        if (kernel.size(-1) % 2) == 0:
            raise ValueError('even sized kernels not yet supported')
        while kernel.dim() < 4: 
            kernel = kernel.unsqueeze(0)  # Make sure kernel has the expected 4 dimensions
        self.kernel = torch.nn.Parameter(kernel,requires_grad=False)
        self.pad = (kernel.size(-1)-1)//2  # Pad image so result will be the same size as the input tensor
        
    def forward(self,tensor):
        return torch.nn.functional.conv2d(tensor,self.kernel,padding=self.pad)

def _test_filters():
    ff = 0.5
#    filt = windowed_gaussian(5)
#    plot_image(filt)

#    print(f'2x2 box {Box2d(2)}')
#    print(f'3x3 box {Box2d(3)}')
#    print(f'4x4 box {Box2d(4)}')
#    print(f'2x2 circbox {Box2d(2,circular=True)}')
#    print(f'3x3 circbox {Box2d(3,circular=True)}')
#    filt = Box2d(100,circular=True)
#    plot_image(filt)

    print(f'2x2 box {Trapezoid2d(2)}')
    print(f'3x3 trap {Trapezoid2d(3)}')
    print(f'4x4 trap {Trapezoid2d(4)}')
    print(f'2x2 circtrap {Trapezoid2d(2,circular=True)}')
    print(f'3x3 circtrap {Trapezoid2d(3,circular=True)}')
    filt = Trapezoid2d(100,full_fraction=ff,circular=False)
    plot_image(filt)
    filt = Trapezoid2d(100,full_fraction=ff,circular=True)
    plot_image(filt)
   
    filt = Trigezoid2d(100,full_fraction=ff,circular=False)
    plot_image(filt)
    filt = Trigezoid2d(100,full_fraction=ff,circular=True)
    plot_image(filt)

    filt = Epanezoid2d(100,full_fraction=ff,circular=False)
    plot_image(filt)
    filt = Epanezoid2d(100,full_fraction=ff,circular=True)
    plot_image(filt)

if __name__ == "__main__":   # execute main() only if run as a script
    _test_filters()      
