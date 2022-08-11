#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:05:26 2019

Code to generate various filters used in the construction of steerable pyramids
The filters are generally defined in Fourier space, but some code is included
to also create limited-footprint spatial approximations suitable for convolution
See Steerable pyramid papers (and my whitepaper) for how the filters are derived and defined

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import torch
import torch.nn.functional as F
import math
from fft_utils import fftshift2d,ifftshift2d, ifft_shim2d


# Generate a pair of high and low pass filters
def create_fourier_high_low_filters_not_yet_used(freq=1,size=255,kerneltype='cos'):
    if kerneltype == 'cos':
        return create_fourier_high_low_filters(freq,size)
    elif kerneltype == 'cos^2':
        [hi,lo] = create_fourier_high_low_filters(freq,size)
        return [hi**2,lo**2]
    else:
        raise ValueError(f'Unrecognized filter type: {kerneltype}')
    
    
# Generate the high and low pass filters for a steerable pyramid
# max_freq is relative to highest frequency (or inverse pixel spacing)
# Note: the default high/low (radial) filters for steerable pyramids use a clamped cosine function in log(r) space
# These are not exactly a partition of frequency space but rather obey the steerable pyramid constraint that low^2 + high^2 = 1 everywhere
def create_fourier_high_low_filters(max_freq=1,size=255):
    if not hasattr(size, "__len__"): size = (size,size)  #make sure size is a 2-tuple (height, width)
    #Create 1d coordinates in normalized space going from about -1 to 1 for each axis except zero must be at an integer coordinate
    center_y = size[0]//2    # y will be zero at this center (should always be at an integer coordinate for FFT)
    y1d = torch.arange(-center_y, size[0]-center_y).float() / center_y
    center_x = size[1]//2
    x1d = torch.arange(-center_x, size[1]-center_x).float() / center_x
    #y1d = torch.linspace(-1,1,steps=size[0])
    #x1d = torch.linspace(-1,1,steps=size[1])
    #Compute distances from center
    x_coord, y_coord = torch.meshgrid(y1d,x1d)
    radius = torch.sqrt(x_coord*x_coord + y_coord*y_coord)
    # Filter is one for frequencies less than max_freq/2
    full = radius.le(max_freq/2).float()
    # Filter smoothly goes to zero for frequencies between max_freq/2 and max_freq
    # weight = cos( pi/2 * log2(2*radius/max_freq) )   with clamping needed to avoid nan from log2(zero)
    partial = ( ((2/max_freq)*radius).clamp(min=0.5).log2()*(math.pi/2) ).cos()
    partial *= radius.gt(max_freq/2).float() * radius.lt(max_freq).float()
    # And is zero for frequenies >= max_freq
    low = partial + full
    # High pass filter is sqrt(1-low*low)
    high = torch.sqrt(1.0 - low*low)
    return [high,low]

# Generate high and low pass filters, similar to steerable pyramid's cosine filters except that the
# filters have a gaussian profile and may extend beyond just one octave
# and these filters do NOT obey the usual pyramid constraint that hi^2 + lo^2 = 1
def create_fourier_gaussian_high_low_filters(freq=1,size=255,sigma=0.65):
    if not hasattr(size, "__len__"): size = (size,size)  #make sure size is a 2-tuple (height, width)
    #Create 1d coordinates in normalized space going from about -1 to 1 for each axis except zero must be at an integer coordinate
    center_y = size[0]//2    # y will be zero at this center (should always be at an integer coordinate for FFT)
    y1d = torch.arange(-center_y, size[0]-center_y).float() / center_y
    center_x = size[1]//2
    x1d = torch.arange(-center_x, size[1]-center_x).float() / center_x
    #Compute distances from center
    x_coord, y_coord = torch.meshgrid(y1d,x1d)
    radius = torch.sqrt(x_coord*x_coord + y_coord*y_coord)
    # compute gaussian bandpass filter first
    log_r = radius.clamp(min=1e-6).log2()
    hi_gauss = (((log_r-math.log2(freq))**2)/(-2*sigma**2)).exp()
    # truncate it to only include the range freq to freq/4
    hi_gauss *= radius.gt(freq/4).float() * radius.lt(freq).float()
    # add back the highest frequencies as one to make it high-pass
    hi_gauss += radius.ge(freq).float()
    
    #now we do a similar thing for the low-pass but shifted up one octave
    lo_gauss = (((log_r-math.log2(freq/2))**2)/(-2*sigma**2)).exp()
    # truncate it to only include the range 2*freq to freq/2
    lo_gauss *= radius.gt(freq/2).float() * radius.lt(2*freq).float()
    # add back the lowest frequencies as one to make it low-pass
    lo_gauss += radius.le(freq/2).float()
    
#    from image_utils import plot_image
#    plot_image(hi_gauss,title=f'hi gauss {size}')
#    plot_image(lo_gauss,title=f'lo gauss {size}')
    return [hi_gauss,lo_gauss]


# Generate the oriented (azimuthal) filters for a steerable pyramid for 4 orientations
# These filters are complex-valued but also purely imaginary so only the imaginary parts are returned (real parts are zero)
# Default version corresponds to complex-valued convolution filters (after inverse fourier transform)
# Steerable version is an odd function and produces to a purely real-valued convolutional filter
def create_fourier_oriented4_imaginary_filters(size=255,steerable=False):
    if not hasattr(size, "__len__"): size = (size,size)  #make sure size is a 2-tuple
    num_orientations = 4      #we do not yet support other numbers of orientations
    #Create 1d coordinates in normalized space going from about -1 to 1 for each axis except zero must be at an integer coordinate
    center_y = size[0]//2    # y will be zero at this center (should always be at an integer coordinate for FFT)
    y1d = torch.arange(-center_y, size[0]-center_y).float() / center_y
    center_x = size[1]//2
    x1d = torch.arange(-center_x, size[1]-center_x).float() / center_x
    #y1d = torch.linspace(-1,1,steps=size[0])
    #x1d = torch.linspace(-1,1,steps=size[1])
    #Compute polar angle theta (about center) for each element
    x_coord, y_coord = torch.meshgrid(y1d,x1d)
    theta = -torch.atan2(x_coord,y_coord)
    #output is set of filters for each orientation we have two tensors for real and imaginary parts
    if steerable:
        magnitude = 2/math.sqrt(5)
    else:
        magnitude = 4/math.sqrt(5)
    filterlist = []
    for orient in range(num_orientations):
        f_imag = magnitude * (torch.cos(theta - (orient*math.pi/4))**3)
        if (not steerable): 
            f_imag = f_imag.clamp(min=0)
        filterlist.append(f_imag)
    return torch.stack(filterlist)   #combine into 4xheightxwidth tensor   

# Generates oriented (azimuthal) filters for an even number of orientations
# as above these filters will be complex-valued and purely imaginary
def create_fourier_oriented_even_imaginary_filters(size=255,steerable=False,*,orientations):
    if not hasattr(size, "__len__"): size = (size,size)  #make sure size is a 2-tuple
    K = orientations
    if (K != int(K)) or (K % 2 != 0): raise ValueError(f"number of orientations must be even {orientations}")
    #Create 1d coordinates in normalized space going from about -1 to 1 for each axis except zero must be at an integer coordinate
    center_y = size[0]//2    # y will be zero at this center (should always be at an integer coordinate for FFT)
    y1d = torch.arange(-center_y, size[0]-center_y).float() / center_y
    center_x = size[1]//2
    x1d = torch.arange(-center_x, size[1]-center_x).float() / center_x
    #y1d = torch.linspace(-1,1,steps=size[0])
    #x1d = torch.linspace(-1,1,steps=size[1])
    #Compute polar angle theta (about center) for each element
    x_coord, y_coord = torch.meshgrid(y1d,x1d)
    theta = -torch.atan2(x_coord,y_coord)
    #output is set of filters for each orientation we have two tensors for real and imaginary parts
    magnitude = (-1**(K/2)) * (2**(K-1)) * math.factorial(K-1) / math.sqrt(K*math.factorial(2*K-2))
    if steerable: 
        magnitude = magnitude/2
    filterlist = []
    for j in range(K):
        f_imag = magnitude * (torch.cos(theta - (j*math.pi/K))**(K-1))
        if (not steerable): 
            f_imag = f_imag.clamp(min=0)
        filterlist.append(f_imag)
    return torch.stack(filterlist)   #combine into Kxheightxwidth tensor   

def create_fourier_oriented_imaginary_filters(size,orientations,steerable=False):
    if orientations==4: 
        return create_fourier_oriented4_imaginary_filters(size,steerable)
    elif orientations%2==0:
        return create_fourier_oriented_even_imaginary_filters(size,steerable,orientations=orientations)
    else:
        raise ValueError(f'Using odd numbers of orientations is not supported {orientations}')
    

# Given a fourier (multiplicative) filter convert it to the corresponding convolutional (spatial) filter
def fourier_to_convolutional_filter(input):
    if input.size(-1) != 2:
        input = torch.stack( (input,torch.zeros_like(input)), dim=-1 )  #assume input was real and add zero imaginary component
    #To perform the inverse fourier transform, we first need to shift from centered representation to one
    #where the origin is the first elelment, then apply ifft and then shift back to centered represenation (origin in middle of domain)
    return fftshift2d(torch.ifft(ifftshift2d(input), 2))   

def fourier_real_to_real_convolutional_filter(input):
    input = torch.stack( (input,torch.zeros_like(input)), dim=-1 )  #assume input was real and add zero imaginary component
    complex = fftshift2d(ifft_shim2d(ifftshift2d(input)))   #compute complex inverse fourier transform
    return complex[:,:,0]      # Return just the real part as 2d tensor

# Given a purely imaginary fourier filter, convert it to the corresponding convolutional filter
def fourier_imaginary_to_convolutional_filter(input):
    input = torch.stack( (torch.zeros_like(input), input), dim=-1 )  #assume input was imaginary and add zero real component
    return fftshift2d(ifft_shim2d(ifftshift2d(input)))   

# Adjust a filter so that its elements sum to the desired target_sum
# This is done by scaling the filters positive and negative values by s and 1/s for some value of s
def adjust_sum_convolutional_filter(input, target_sum=0):
    # Separate filter into its positive and negative elements
    pos = input.clamp(min=0)
    neg = input.clamp(max=0)
    p = pos.sum()
    n = neg.sum()
    s = target_sum
    # Solve fot scale factor s such that pos*s + neg/s == target_sum
    scale = (s + math.sqrt(s*s - 4*n*p)) / (2*p)
    res = scale*pos + neg/scale
    #print(input.size())
    #print(f"sums old:{input.sum()} new: {res.sum()} scale:{scale}")
    return res
    
# Given a convolutional filter, try to crop out a smaller filter with roughly the same norm
# Very useful if the filters are surrouded by large regions with nearly zero values
def trim_convolutional_filter(input,threshold=0.98,dim=(0,1)):
    fullnorm = input.norm()
    fullsum = input.sum().item()
    sumthreshold = 0.05*max(fullsum,1.0)
    if (input.dim()>2): sumthreshold *= input.size(0)  #if it is a stack of filter raise threshold
#    print(fullsum, sumthreshold)
    #assume filter is square for now
    center = input.size(dim[0])//2
    radius = 1
    #clip out a central portion of filter and test if its norm is close enough
    while radius <= center:
        test = input.narrow(dim[0],center-radius,2*radius+1).narrow(dim[1],center-radius,2*radius+1)
#        print(abs(test.sum().item()-fullsum))
        if (test.norm() >= threshold*fullnorm) and (abs(test.sum().item()-fullsum) <= sumthreshold):
            break
        radius += 1
    trim = test.clone().detach()   #return copy of trimmed tensor (so it does not share backing store)
    if input.dim()==2:
        trim = adjust_sum_convolutional_filter(trim,fullsum)
    elif input.dim()==4:
        for i in range(input.size(0)):
            for j in range(input.size(3)):
                filt = trim[i,:,:,j]
                trim[i,:,:,j] = adjust_sum_convolutional_filter(filt,input[i,:,:,j].sum().item())
    else:
        print(f"In trim_convolutional_filter() sum adjustment not yet imprlemented for: {trim.size()}")
    #print(f"trim sum old:{fullsum} new:{trim.sum().item()}")
    return trim


# Apply a convolutional (really correlational) filter to an image to produce output with same size as input image
def conv2d_keep_size(image,filter):
    pad = (filter.size(-1)-1)//2   #we need to add padding so convolution does not change its size
    # If input is not in batch x channel x height x width form, add trivial batch dimensions of length 1
    if image.dim()==3: image = image.unsqueeze(0)
    if filter.dim()==3: filter = filter.unsqueeze(0)
    res = F.conv2d(image,filter,padding=pad)
    return res

def __test():
    (hi,lo) = create_fourier_high_low_filters(max_freq=1,size=4)
    #print(lo)
    print(ifftshift2d(lo))
    #print(ifftshift2d(hi))
    
    
if __name__ == "__main__":   # execute main() only if run as a script
    __test()

