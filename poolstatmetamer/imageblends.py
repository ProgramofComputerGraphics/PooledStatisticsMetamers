#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:22:44 2019

Routines for compositing, blending, or overlaying images (stored as torch.Tensors)
and (using ffmpeg) for converting images into movies

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import math
import torch
import subprocess
import shutil
import torchvision.transforms
from PIL import Image, ImageDraw, ImageFont
from image_utils import plot_image

import os
import numpy as np

# Compute an image that is the same size as the input but where each pixel contains the distance
# to the specified center point (default is middle of image)
# Useful for a variety of kernel construction and blending purposes
# Note center is in torch tensor coordinates (so y comes before x)
# Step is the spacing between elements in the return distance image
#  if step is not 1, then return image size will be different than the input image size
# Can optionally also return the dx and dy images by setting return_dx_dy to true
def distance_image(image_size,center=None,step=1,return_squared_distance=False,return_dx_dy=False):
    if torch.is_tensor(image_size):
        width = image_size.size(-1)
        height = image_size.size(-2)
    elif isinstance(image_size,(torch.Size,tuple,list)):
        width = image_size[-1]
        height = image_size[-2]  # assume tuple is in torch ordering (...,height,width)
    elif isinstance(image_size,(int,float)):
        width = image_size
        height = image_size
    if center is None: 
        center = ((height-1)/2,(width-1)/2)  # if center not specified use middle of image
    x = torch.arange(0,width,step=step).float() - center[-1]
    y = torch.arange(0,height,step=step).float() - center[-2]
    x2 = (x**2).unsqueeze(0)
    y2 = (y**2).unsqueeze(1)
    dist2 = x2 + y2
    if return_squared_distance:
        retval = dist2      # can optionally return distance squared from center if requested
    else:
        retval = torch.sqrt(dist2)  # default is to return distance from center 
    if return_dx_dy: return retval,x.unsqueeze(0).expand(dist2.size()),y.unsqueeze(1).expand(dist2.size())
    return retval    

# Kernels for blending images, kernels are same size as the target images but with configurable centers

# Kernel which is one inside the inner radius and blends smoothly to zero at the outer_radius using cosine^2
# A 2D radially-symmetric kernel
def trapezoid_radial(image,inner_radius,outer_radius=None,center=None):
    if outer_radius is None: outer_radius = inner_radius*2
    dist = distance_image(image,center=center)
    blend_func = (outer_radius - dist)/(outer_radius - inner_radius)
    blend_func.clamp_(min=0,max=1)
    return blend_func

# Kernel which is one inside the inner radius and blends smoothly to zero at the outer_radius using cosine^2
# A 2D radially-symmetric kernel
def trigezoid_radial(image,inner_radius,outer_radius=None,center=None):
    if outer_radius is None: outer_radius = inner_radius*2
    dist = distance_image(image,center=center)
    angles = (dist - inner_radius)*(math.pi/(2*(outer_radius-inner_radius)))
    angles.clamp_(min=0)
    blend_func = (torch.cos(angles)**2) * dist.lt(outer_radius).float()
    return blend_func

# Blend two images according the specified weights (for the first image)
def blend_images(imageA, imageB, weightA):
    weightA = weightA.to(imageA)   # Convert weight to same type as image (eg byte to float)
    #return torch.lerp(imageA,imageB,weightA)   lerp with tensor weights was only recently added and not yet supported in my version of pytorch
    return imageA*weightA + imageB*(1-weightA)

# Blend a list of images according to provided radii (which should be in increasing order)
# Last entry can be just a bare tensor for the background image
# For example radiallist = [(foreground,inner,outer),...,background]
def radial_blend_images(center,radiallist):
    cur = None
    for raddesc in reversed(radiallist):
        if torch.is_tensor(raddesc):  # if it is a bare tensor, then set current to it
            cur = raddesc
            #print(f'cur {cur.size()}')
        else:                         # else assume its a tuple (img,inner_radius<,outer_radius>)
            img = raddesc[0]
            inner_radius = raddesc[1]
            outer_radius = raddesc[2] if len(raddesc)>2 else None
            #print(f'img {img.size()} inner_radius {inner_radius} outer_radius {outer_radius}')
            kern = trapezoid_radial(cur,inner_radius,outer_radius,center=center).unsqueeze(0)
            if cur is None: cur = torch.zeros_like(img)  # handle special case if no previous or background image then set to black
            cur = blend_images(img,cur,kern)
    return cur

# Utility to invoke ffmpeg (external program) to compile frames into a movie file
def compile_frames_to_mp4(inputpattern='frame%03d.png',outputfilename='out.mp4',codec='libx264',vlc_only=False,verbose=False,framerate=None):
    if not shutil.which('ffmpeg'):
        raise FileNotFoundError('ffmpeg executable not found.  Make sure it is installed on this system (eg by running "conda install ffmpeg")')
    if framerate is None: framerate = 30  # A reasonable default if no framerate was specified
    # Run ffmpeg using commandline to compile frames to mp4 movie
    cmd = ['ffmpeg', 
           '-y',                        # allow overwriting movie file if it already exists
           '-framerate', f'{framerate}',# set the framerate (for input) which output should also inherit
           '-i', inputpattern,       # filename pattern, used to find frame image files
           '-vcodec', codec]         # x265 codec produces much better output than the default codec
    if not vlc_only:
        cmd.append('-pix_fmt')
        cmd.append('yuv420p')    # use chroma subsampling required by many video players including quicktime (VLC does not require this)    
    cmd.append(outputfilename)
    # execute command, check return value is not an error
    if verbose:
        subprocess.check_output(cmd)  # prints command output and errors to terminal
    else:
        try:
            subprocess.run(cmd,check=True,capture_output=True)  # run command silently, but check return value for errors
        except subprocess.CalledProcessError as e:
            print('Process stderr: '+e.stderr.decode())
            raise e
    print(f'movie compiled to {outputfilename}')

    
def compile_convergence_graph_cv2(iterdir, framenames, framerate=30):
    import cv2  # place here so openCV will only be required if this routine is used (not currently used elsewhere)
    bsdir = os.path.dirname(iterdir)
    out_video_name=os.path.join(bsdir,'convergence_graph.avi')
    height, width, channels = np.shape(cv2.imread(os.path.join(iterdir, f'metamer_iter000.png')))
    video = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc(*'XVID'), framerate, (width,height))
    i=0
    print(os.path.join(iterdir, f'metamer_iter000.png'))
    frame = cv2.imread(os.path.join(iterdir, f'metamer_iter000.png'))
    while(frame is not None):
        video.write(frame)
        i+=1
        frame = cv2.imread(os.path.join(iterdir, f'metamer_iter{str(i).zfill(3)}.png'))
    video.release()
    print(f'movie compiled to {out_video_name}')

# Given a string, rasterize it into an image sized to fit the text
# Fonts are annoying platform and install specific, but you can easily use the (low quality) default bitmap font
def text_to_image(text,font=None):
    if font is None:
        font = ImageFont.load_default()     # If no font was provided, use the simple default bitmap one
#    elif isinstance(font,str):
#        font = ImageFont.truetype(font, 12)  # UGH! Font loading involves platform specific paths (and font names)
    size = font.getsize(text)          # size of text in pixels (width,height)
    pimg = Image.new('L',size)
    draw = ImageDraw.Draw(pimg)
    draw.text((0,0),text,255,font=font)
    tensor = torchvision.transforms.ToTensor()(pimg)
    return tensor

# Insert (overlay) a small image within a larger image
# Warning: operation happens in place and will overwrite parts of the input tensor
def overlay_subimage_(tensor,subimage,op='add'):
    width = subimage.size(-1)
    height = subimage.size(-2)
    spot = tensor.narrow(-1,tensor.size(-1)-width-2,width).narrow(-2,tensor.size(-2)-height-1,height)
    if op=='add':
        spot.add_(subimage)
    elif op=='set':
        spot.copy_(subimage)
    else:
        raise ValueError(f"unrecognized overlap operation {op}")
    return tensor

# Draw some text into the lower right hand corner of an image
def overdraw_text(tensor,text,color=1):
    timg = text_to_image(text)*color    # Convert text string to an image
    tensor = tensor.clone()             # Make a copy of image so we don't overwrite the original
    overlay_subimage_(tensor,timg)      # Overlay text onto source image
    return tensor

def _test_blend():
    img = torch.zeros(64,64)
    kern = trigezoid_radial(img,16,32,center=(16,48))
    plot_image(kern)
    plot_image(blend_images(torch.ones_like(img),img,kern))
    
def _test_textoverlay():
    #img = _text_to_tensor("1234")
    img = torch.zeros(1,256,256)
    img = overdraw_text(img,"1234",0.5)
    #show_image(img)
    plot_image(img)

    
if __name__ == "__main__":   # execute main() only if run as a script
    _test_blend()      
    _test_textoverlay()
