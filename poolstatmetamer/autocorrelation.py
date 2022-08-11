#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:00:39 2019

Utilities for computing autocorrelations for images stored as torch.Tensors
Computes autocorrelation for a single specified offset vector and assumes
 zero boundary condition (image is assumed to be zero outside of its boundaries)

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import torch

# This version formats the result as an image in pytorch's preferred 4d format but accepts 2d, 3d, or 4d images
def autocorrelation2d(stack, offset, offset_scale=1, center=True):
    """
    Computes autocorrelation image for a tensor and a single fixed offset
    
    Autocorrelation image is the product of the image times an offset, or shifted, version of itself
    """
    #convert input images to pytorch standard 4d format if needed (batch,channel,height,width)
    if stack.dim()==2:
        stack = stack.unsqueeze(0).unsqueeze(0)
    elif stack.dim()==3:
        stack = stack.unsqueeze(0)
    res = autocorrelation_image(stack,offset,offset_scale,center)
    return res
        
# Given an image (in standard 4d format) and an offset (in 2d), return the product of the image and a offset/shifted version of itself
# Note: the offset refers to the last two dimensions only (we don't support offsets in image batch number or channel)
def autocorrelation_image(image, offset, offset_scale=1, center=True):
    """
    Computes the product of an image times offset copy of itself.
    
    Pixels outside of the image are assumed to be zero, so regions where the offset
    extends beyond the image are set to zero.
    
    Args:
        image: input image
        offset: amount to shift image before multplying it by the unshifted image
        offset_scale: multiplier applied to offset before it is used 
        center: shift result by half the offset (as if half the offset was applied to each image in opposite directions)
    Return:
        Product image of same size as input
    """
    #print(f"autocorrelation size: {image.size()} offset: {offset} scale:{offset_scale}")
    if len(offset)!=2: raise ValueError(f'Expected 2 dimensional offset but got: {offset}')
    if image.dim() != 4: raise ValueError(f'Expected 4 dimensional image but got: {image.size()}')
    res = torch.zeros_like(image)
    d0 = offset[0]*offset_scale
    d1 = offset[1]*offset_scale
    s0 = image.size(-2)
    s1 = image.size(-1)
    if abs(d0) >= s0: return res  #if no overlap with shifted version, just return all zeros result
    if abs(d1) >= s1: return res
    # centering shifts the results to middle of offset vector rather than start
    # centering makes shift and -shift produce exactly the same result
    if center:
        c0 = abs(d0//2)
        c1 = abs(d1//2)
    else:
        c0 = 0
        c1 = 0
    # Compute the product of the image and its shifted version and in the region where they overlap
    if d0 >= 0:
        if d1 >= 0:
            res[:,:,c0:c0+s0-d0,c1:c1+s1-d1] = image[:,:,:s0-d0,:s1-d1] * image[:,:,d0:,d1:]
        else:
            res[:,:,c0:c0+s0-d0,-c1-d1:s1-c1] = image[:,:,:s0-d0,-d1:] * image[:,:,d0:,:s1+d1]
    else:
        if d1 >= 0:
            res[:,:,-c0-d0:s0-c0,c1:c1+s1-d1] = image[:,:,-d0:,:s1-d1] * image[:,:,:s0+d0,d1:]
        else:
            res[:,:,-c0-d0:s0-c0,-c1-d1:s1-c1] = image[:,:,-d0:,-d1:] * image[:,:,:s0+d0,:s1+d1]   
    return res

# This version formats the result as an image in pytorch's preferred 4d format
def offsetcorrelation2d(imageA,imageB, offset, offset_scale=1, center=True):
    """
    Computes offset correlation image for two tensors and a single fixed offset
    
    Offsetcorrelation image is the product of the image times an offset, or shifted, version of the second image
    """
    #convert input images to pytorch standard 4d format if needed (batch,channel,height,width)
    if imageA.dim()==2:
        imageA = imageA.unsqueeze(0).unsqueeze(0)
    elif imageA.dim()==3:
        imageA = imageA.unsqueeze(0)
    if imageB.dim()==2:
        imageB = imageB.unsqueeze(0).unsqueeze(0)
    elif imageB.dim()==3:
        imageB = imageB.unsqueeze(0)
    res = offsetcorrelation_image(imageA,imageB,offset,offset_scale,center)
    return res
        
# Given two 2d images and an offset, return the product of the image and a offset/shifted imageB
def offsetcorrelation_image(imageA, imageB, offset, offset_scale=1, center=True):
    """
    Computes the product of an imageA times an offset version of imageB.
    
    Pixels outside of the images are assumed to be zero, so regions where the offset
    extends beyond the image are set to zero.
    
    Args:
        imageA: first input image
        imageB: second input image
        offset: amount to shift imageB before multplying it by imageA
        offset_scale: multiplier applied to offset before it is used 
    Return:
        Product image of same size as input
    """
    #print(f"image correlation size: {imageA.size()} offset: {offset} scale:{offset_scale}")
    if len(offset)!=2: raise ValueError(f'Expected 2 dimensional offset but got: {offset}')
    if imageA.size() != imageB.size(): raise ValueError(f'Expected images with matching size: {imageA.size()} vs {imageB.size()}')
    if imageA.dim() != 4: raise ValueError(f'Expected 4 dimensional image but got: {imageA.size()}')
    res = torch.zeros_like(imageA)
    d0 = offset[0]*offset_scale
    d1 = offset[1]*offset_scale
    s0 = imageA.size(-2)
    s1 = imageB.size(-1)
    if abs(d0) >= s0: return res  #if no overlap with shifted version, just return all zeros result
    if abs(d1) >= s1: return res
    # centering shifts the results to middle of offset vector rather than start
    # centering makes shift and -shift produce exactly the same result
    if center:
        c0 = abs(d0//2)
        c1 = abs(d1//2)
    else:
        c0 = 0
        c1 = 0
    # Compute the product of the image and its shifted version and in the region where they overlap
    if d0 >= 0:
        if d1 >= 0:
            res[:,:,c0:c0+s0-d0,c1:c1+s1-d1] = imageA[:,:,:s0-d0,:s1-d1] * imageB[:,:,d0:,d1:]
        else:
            res[:,:,c0:c0+s0-d0,-c1-d1:s1-c1] = imageA[:,:,:s0-d0,-d1:] * imageB[:,:,d0:,:s1+d1]
    else:
        if d1 >= 0:
            res[:,:,-c0-d0:s0-c0,c1:c1+s1-d1] = imageA[:,:,-d0:,:s1-d1] * imageB[:,:,:s0+d0,d1:]
        else:
            res[:,:,-c0-d0:s0-c0,-c1-d1:s1-c1] = imageA[:,:,-d0:,-d1:] * imageB[:,:,:s0+d0,:s1+d1]   
    return res

def generate_offset_list(window_size=7,include_zero_offset=False):
    """ 
    Generates a list of unique offsets within the specified window size.
    
    List will includes offsets within the specified window for use in autocorrelation computations.
    Window size must be odd (so 0,0 is the center of the window)
    By default the list does not include null shift (0,0) 
    or redundant values such as (a,b) and (-a,-b), since
    autocorrelation is assumed to be symmetric with respect to shifts
    """
    radius = window_size//2
    if window_size != 2*radius+1 : raise Exception("invalid window size")
    res = []
    if include_zero_offset: res.append( (0,0) )
    # Note we only include offsets where first number is >= 0 to avoid redundant computation
    # Negating the offset would compute the same autocorrelation (just shifted by a few pixels)
    for j in range(1,radius+1):
        res.append( (0,j) )
    for i in range(1,radius+1):
        for j in range(-radius,radius+1):
            res.append( (i,j) )
    return res

    
offset_exactly_2_list = ((0,2),(1,-2),(1,2),(2,-2),(2,-1),(2,0),(2,1),(2,2))
offset_8neighbors_list = ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
offset_8neighbors_radius2_list = ((2,0),(2,2),(0,2),(-2,2),(-2,0),(-2,-2),(0,-2),(2,-2))
def generate_fixed_max_offset_list(radius):
    """
    Generates a list of unique offsets where one value is equal to the radius and the absolute value of the other is <= radius
    
    List is set of offsets at a particular radius in the L-infinity or max metric.
    This is equivalent to the offsets for a window_size of 2*radius+1 - those from
    a window size of 2*radius-1
    equivalently the offsets for that windows size minus the offsets from a window
    size of one less
    """
    if radius < 1 : raise Exception(f"invalid radius {radius}")
    res = []
    # Note we only include offsets where first number is >= 0 to avoid redundant computation
    # Negating the offset would compute the same autocorrelation (just shifted by a few pixels)
    res.append( (0,radius) )
    for i in range(1,radius):
        res.append( (i,-radius) )
        res.append( (i,radius) )
    for j in range(-radius,radius+1):
        res.append( (radius,j) )
    return res
    


def __test():
    print(f"Offsets for window=3: {generate_offset_list(3)}")
    print(f"Number of offsets for window=7 is {len(generate_offset_list(7))}")
    print(f"shifts of radius 2: {generate_fixed_max_offset_list(2)}")
    a = torch.arange(1,10).view(3,3)
    shift = (1,1)
    print(a)
    print(f"after autocorrelation with shift {shift} without recentering")
    print(autocorrelation2d(a, shift, center=False))
    #print(autocorrelation2d(a, shift, center=True))
    b = a.repeat((2,2))
    print(b)
    shift = (2,-0)
    print(f"after autocorrelation with shift {shift}")
    print(autocorrelation2d(b, shift, center=False))
    print(autocorrelation2d(b, shift, center=True))
    
if __name__ == "__main__":   # execute main() only if run as a script
    __test()

