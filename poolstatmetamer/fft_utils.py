#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:29:35 2019

Some useful utility routines for processing and handling discrete fourier transforms (ie FFTs)
Pytorch often represents complex numbers by having a final dimension of length 2 (for real and imaginary components)

Note: pytorch changed to a new fft API in pytorch 1.8 (torch.fft.*) that is not backwards compatible 
and removed the older fft interface that we had been using.  For now I'm putting simple shim functions to allow 
our code to work with either FFT API.  Eventually these should likely be removed in favor of directly calling 
the new fft API (with its complex-value tensors).

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import sys
import torch

# Utility routine for performing fftshift and similar transforms that rotate the elements of a tensor
def _fftshift(input,dimensions,offset):
    for dim in dimensions:
        size = input.size(dim)
        split = (size+offset)//2
        left = input.narrow(dim,0,split)
        right = input.narrow(dim,split,size-split)
        input = torch.cat( (right,left), dim)
    return input

# Rotate first element (eg lowest frequency) to middle of the image tensor (rather than at the left-edge as is the default for FFT routines)
def fftshift2d(input):
    """
    Performs fftshift on 2d tensor (real or complex).  Shifts DC component to middle position.
    """
    # This could be made a bit more efficient by doing both dimensions at the same time
    if (input.dim()>2) and (input.size(-1)==2):
        return _fftshift(input,(-2,-3),1) #assume its a complex tensor with last dimension being real,imaginary parts
    else:
        return _fftshift(input,(-1,-2),1) #assume its a real tensor

# Inverse of fftshift2d.  Rotates the middle element to first element in tensor 
def ifftshift2d(input):
    """
    Performs ifftshfit on 2d tensor (real or complex).  Inverse of fftshift2d().
    """
    if (input.dim()>2) and (input.size(-1)==2):
        return _fftshift(input,(-2,-3),0) #assume its a complex tensor
    else:
        return _fftshift(input,(-1,-2),0) #assume its a real tensor

# Remove the high-frequency (middle) elements from an FFT result 
# Useful if the FFT has been low-pass filtered so that these frequencies are known to be zero
# Equivalent to performaing an decimation operation in the spatial domain
def _freq_downsample(input,dimensions,factor):
    for dim in dimensions:
        insize = input.size(dim)
        outsize = insize//factor
        # The downsample factor divide evenly into the image size
        if insize != (factor*outsize): raise RuntimeError(f'bad downsample size {insize} is not a multiple of {factor}')
        rsize = outsize//2
        lsize = outsize - rsize
        left = input.narrow(dim,0,lsize)
        right = input.narrow(dim,insize-rsize,rsize)
        input = torch.cat((left,right), dim)    
    return input

# Remove the high-frequency (middle) elements from an FFT result 
def _freq_downsample2d(input,dim1,dim2,factor):
        s1 = input.size(dim1)//factor
        s2 = input.size(dim2)//factor
        if s1*factor!=input.size(dim1) :
            raise RuntimeError(f'invalid image size in freq_downsample: {input.size(dim1)} is not a multiple of {factor}')
        if s2*factor!=input.size(dim2):    
            raise RuntimeError(f'invalid image size in freq_downsample: {input.size(dim2)} is not a multiple of {factor}')
        # Extract the four image corners we want to keep using narrow operations (does not copy tensor data)
        r1 = s1//2
        r2 = s2//2
        l1 = s1 - r1
        l2 = s2 - r2
        img_l = input.narrow(dim1,0,l1)
        img_r = input.narrow(dim1,-r1,r1)
        img_ll = img_l.narrow(dim2,0,l2)
        img_lr = img_l.narrow(dim2,-r2,r2)
        img_rl = img_r.narrow(dim2,0,l2)
        img_rr = img_r.narrow(dim2,-r2,r2)
        # Now assemble the four corners by concatenation to form a new smaller matrix
        output = torch.cat((torch.cat((img_ll,img_lr),dim2), torch.cat((img_rl,img_rr),dim2)), dim1)
        return output
    
# Shrink frequency image by removing the highest frequencies from an FFT result and return the reduced size result
# Assumes zero frequency is at (0,0) in frequency image (ie fftshift has not been applied)
# Equivalent to performing a decimation operation in the spatial domain
def freq_downsample2d(freqimage,factor):
    if (factor<1): raise RuntimeError(f"downsampling factor cannot be less than 1: {factor}")
    if (factor==1): return freqimage   # Size is unchanged 
    if (freqimage.dim()>2) and (freqimage.size(-1)==2):
        return _freq_downsample(freqimage,(-2,-3),factor) #assume its a complex tensor where last dimension is real,imag
    else:
        return _freq_downsample2d(freqimage,-1,-2,factor)
#        return _freq_downsample(freqimage,(-1,-2),factor) #assume its a real tensor

# Expand frequency image by adding higher frequencies (with zero values) and return the expanded frequency image
# Inverse of freq_downsample2d if the original was properly low pass filtered first
# Assumes zero frequency is at (0,0) in frequency image (ie fftshift has not been applied)
def freq_upsample2d(freqimage,factor):
    if (factor<1): raise RuntimeError(f"upsampling factor cannot be less than 1: {factor}")
    if (factor==1): return freqimage   # Size is unchanged
    # Note eventually this could be implemented more efficiently without the fftshift 
    img = fftshift2d(freqimage)
    if (freqimage.dim()>2) and (freqimage.size(-1)==2):
        #assume its a complex tensor where last dimension is real,imag
#        torch.nn.functional.pad(img,(0,0))
        p1 = (factor-1)*img.size(-2)
        p2 = (factor-1)*img.size(-3)
        img = torch.nn.functional.pad(img,(0,0, (p1+1)//2,(p1+0)//2, (p2+1)//2,(p2+0)//2))
    else:
        #assume its a real tensor
        p1 = (factor-1)*img.size(-1)
        p2 = (factor-1)*img.size(-2)
        img = torch.nn.functional.pad(img,((p1+1)//2,(p1+0)//2, (p2+1)//2,(p2+0)//2))
    img = ifftshift2d(img)
    return img;
    
#-----FFT-Migration-Shim-Functions----------------------------------

def rfft_shim2d(image):
    if "torch.fft" not in sys.modules:
        return torch.rfft(image,2,onesided=False)
    else:
        return torch.view_as_real(torch.fft.fft2(image))
    
def irfft_shim2d(image):
    if "torch.fft" not in sys.modules:
        return torch.irfft(image, 2, onesided=False)
    else:
        return torch.fft.ifft2(torch.view_as_complex(image)).real
    
def ifft_shim2d(image):
    if "torch.fft" not in sys.modules:
        return torch.ifft(image, 2)   
    else:
        return torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(image)))
    
        
        
def __test():
    '''
    (hi,lo) = create_fourier_high_low_filters(max_freq=1,size=4)
    lo = ifftshift2d(lo)
    print(lo)
    print(fftdownsample2d(lo,2))
    #print(ifftshift2d(hi))
    '''
    s = 8
    a = torch.arange(1,s*s+1).view(s,s)
    print(a)
    d = freq_downsample2d(a,2)
    print(d)
    print(freq_upsample2d(d,2))
    
def __test_downsample():
    from image_utils import plot_image,plot_images
    import spyramid_filters as spf
    s = 8
    off = 0
#    img = torch.zeros(4*s,2*s)
    img = torch.zeros(1*s,1*s)
#    img[off:off+s,off:off+s] = 1
    img[0,0] = 1
    #print(img)
#    freq = torch.rfft(img,2,onesided=False)
    freq = rfft_shim2d(img)
    #print(freq[:,:,0])
    #print(freq[:,:,1])
    plot_images([img,freq[:,:,0],freq[:,:,1]])
    [hifilt, lofilt] = spf.create_fourier_high_low_filters(size=img.size(),max_freq=0.5)
    hifilt = ifftshift2d(hifilt)
    lofilt = ifftshift2d(lofilt)
    plot_images([hifilt,lofilt],title="hi and lo filters")
    lofreq = freq*lofilt.unsqueeze(-1)
    loimg = irfft_shim2d(lofreq)
    plot_images([loimg,lofreq[:,:,0],lofreq[:,:,1]],title="low-pass image, freq-real, freq-imag")
    downfreq = freq_downsample2d(lofreq,2)
    downimg = irfft_shim2d(downfreq)
    plot_images([downimg,downfreq[:,:,0]],title="downsampled-low-pass and real-freq")
    upfreq = freq_upsample2d(downfreq,2)
    plot_images([lofreq[:,:,0],upfreq[:,:,0]])
    print(torch.equal(upfreq,lofreq))
    
    
    
if __name__ == "__main__":   # execute main() only if run as a script
    #__test()
    __test_downsample()


