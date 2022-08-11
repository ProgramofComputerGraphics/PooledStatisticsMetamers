#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:05:03 2021

Utilities for computing auto-differences for images stored as torch.Tensors
The auto-difference is equal to an image minus an translated version of that image
  The translation is defined by the offset vector.  If the subtraction would involve
  a pixel outside the boundary of the original image, then its result is set to zero
Our auto-difference operations are analogous to autocorrelations (except using 
  subtraction instead of multiplication) and are used in our experimental edge-stop statistics

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import torch

# This version formats the result as an image in pytorch's preferred 4d format but accepts 2d, 3d, or 4d images
def autodifference2d(stack, offset, offset_scale=1, center=True):
    """
    Computes autodifference image for a tensor and a single fixed offset
    
    Offsetdiff image is the difference of the image times an offset, or shifted, version of itself
    """
    #convert input images to pytorch standard 4d format if needed (batch,channel,height,width)
    if stack.dim()==2:
        stack = stack.unsqueeze(0).unsqueeze(0)
    elif stack.dim()==3:
        stack = stack.unsqueeze(0)
    res = autodifference_image(stack,offset,offset_scale,center)
    return res

# Given an image (in 4d format) and an offset (in 2d), return the difference of the image and a offset/shifted version of itself
# in places where the offset would be outside of the image, the result will be set to zero
# Note: the offset refers to the last two dimensions only (we don't support offsets in image batch number or channel)
def autodifference_image(image, offset, offset_scale=1, center=True):
    """
    Computes the result of an image minus an offset copy of itself.
    
    Pixels outside of the image are undefined and we set any results depending on them to zero
    
    Args:
        image: input image
        offset: amount to shift image before subtracting it from the unshifted image
        offset_scale: multiplier applied to offset before it is used 
        center: shift the result image by half the offset (as if half the offset was applied to each image copy but in opposite directions)
    Return:
        Difference image of same size as input
    """
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
    # Compute the difference of the image and its shifted version and in the region where they overlap
    if d0 >= 0:
        if d1 >= 0:
            res[:,:,c0:c0+s0-d0,c1:c1+s1-d1] = image[:,:,:s0-d0,:s1-d1] - image[:,:,d0:,d1:]
        else:
            res[:,:,c0:c0+s0-d0,-c1-d1:s1-c1] = image[:,:,:s0-d0,-d1:] - image[:,:,d0:,:s1+d1]
    else:
        if d1 >= 0:
            res[:,:,-c0-d0:s0-c0,c1:c1+s1-d1] = image[:,:,-d0:,:s1-d1] - image[:,:,:s0+d0,d1:]
        else:
            res[:,:,-c0-d0:s0-c0,-c1-d1:s1-c1] = image[:,:,-d0:,-d1:] - image[:,:,:s0+d0,:s1+d1]   
    return res

def __test():
    a = torch.arange(1,10).view(3,3)
    shift = (1,1)
    print(a)
    print(f"after autodifference with shift {shift} without recentering")
    print(autodifference2d(a, shift, center=False))
    #print(autocorrelation2d(a, shift, center=True))
    b = a.repeat((2,2))
    print(b)
    shift = (2,-0)
    print(f"after autodifference with shift {shift}")
    print(autodifference2d(b, shift, center=False))
    print(autodifference2d(b, shift, center=True))
    
if __name__ == "__main__":   # execute main() only if run as a script
    __test()