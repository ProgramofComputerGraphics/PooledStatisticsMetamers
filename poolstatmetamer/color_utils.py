#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:39:46 2019

Some useful color transforms. Mainly an approximate transform for LMS cone space
(long, medium, & short wavelength cones) and an associated opponent color space.
In future we could add methods for computing LMS from monitor primaries.

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import torch

# A approximate transform from a typical RGB space to cone LMS space
# Would be good to check this to see if is a reasonable default approximation for 
# human cone LMS, but results so far don't seem to depend much on precise color transform details
rgb2lms = torch.tensor([[0.3811, 0.5783, 0.0402],
                        [0.1967, 0.7244, 0.0782],
                        [0.0241, 0.1288, 0.8444]])

lms2rgb = rgb2lms.inverse()

# A simple approximation of the opponent cone color space (achromatic,red-green,blue-yellow)
# I also scaled their magnitudes to make the channels have more simlar ranges
lms2opc = torch.tensor([[0.5, 0.5, 0],    # (L+M) / 2)
                       [-4, 4, 0],        # (M-L) * 3
                       [0.5, 0.5, -1]])  # (L+M)/2 - S)

opc2lms = lms2opc.inverse()

# Composite transform from RGB to cone-opponent color space
rgb2opc = torch.matmul(lms2opc,rgb2lms)

opc2rgb = rgb2opc.inverse()

# Short names for the three opponent channels, short for (achromatic,red-green,blue-yellow)
opc_short_names = ('ac','rg','by')


# Apply the specify color tranformation matrix to an image
def color_transform_image(image,color_matrix):
    if image.dim()==3:
        return torch.nn.functional.conv2d(image[None,:,:,:],color_matrix[:,:,None,None])
    else:
        return torch.nn.functional.conv2d(image,color_matrix[:,:,None,None])
    
# Given a color represented as a list (of channel values) apply color matrix and return the result as a list
def color_transform_list(colorlist,color_matrix):
    before = torch.tensor(colorlist,dtype=torch.float)
    after = torch.mv(color_matrix,before)
    return after.tolist()
    
# Convert an RGB image to cone LMS image
def rgb_to_coneLMS(image):
    return color_transform_image(image,rgb2lms)

# Convert an RGB image to a cone opponent space image
def rgb_to_opponentcone(image):
    return color_transform_image(image,rgb2opc)


# Version encoded as a torch.nn.Module (useful for inserting into trainable networks
#  and automatically supporting GPU acceleration)
class ColorTransform(torch.nn.Module):
    
    def __init__(self,color_matrix,channel_names):
        super().__init__()
        self.matrix = torch.nn.Parameter(color_matrix,requires_grad=False)
        self._short_channel_names = channel_names
        
    def forward(self,image):
        return color_transform_image(image,self.matrix)
    
    def channel_names(self):
        return self._short_channel_names

""#END-CLASS------------------------------------

# Create a color tranform module for RGB to opponentcone color spaces
def RGBToOpponentConeTransform():
    return ColorTransform(rgb2opc,opc_short_names)


def _test():
    from image_utils import load_image_rgb, plot_image, plot_image_channels

    print(rgb2lms)
    print(lms2opc)
    print(rgb2opc)
    image = load_image_rgb('../sampleimages/cat256.png')
    plot_image(image)
    imageB = rgb_to_opponentcone(image)
    print(imageB.size())
    #plot_image(imageB.squeeze())
    plot_image_channels(imageB)
    print(lms2opc @ lms2opc.t())
    print(rgb2opc @ rgb2opc.t())
    print(color_transform_list((1,1,1), rgb2opc))
    

if __name__ == "__main__":   # execute main() only if run as a script
    _test()       
