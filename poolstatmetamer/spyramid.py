#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:09:59 2019

Steerable pyramids and some methods for computing them from an image

Update: We have further generalized the idea of steerable pyramids to allow
more flexibility over which types of images are stored for each frequency
range.  The wavelength (1/frequency) space is divided into a fixed number
of bands starting with level 0 (highest frequencies) up to some maximum level
(lowest frequencies).  Each level can optionally store a band-pass image, a
low-pass image, and/or a set of oriented edge images.  Images are stored in 
lists indexed by level (missing images are indicated by the value None).
The frequency bands use soft boundaries and neighboring bands overlap in
frequency space.  The zero level is limited by the image resolutions maximum
representable frequency rather than an explicit high-pass filter (and covers
a slightly larger range)

SPyramid class represents an (extended) steerable pyramid image decomposition.
A steerable pyramid consists of an original/base image and a set of linearly 
filtered images derived from it at various scales and orientations (consisting 
of high-pass, low-pass, and edge images).  The number of levels in the pyramid 
can be configured but the number of orientations is currently fixed at 4.

These extended steerable pyramids contain some additional types of edge images.
Besides the real (aka even) edge images, it also contains imaginary (aka odd) 
edge images; collectively, these are known as edge *phase* images and contain
both positive and negative components.  Two types of non-linearly filtered
images are also generated.  Phase-doubled images are phase images that 
are modified to oscillate at twice their original frequency so that 
they match the phase frequency of the next finer scale.
Edge magnitude images are strictly positive-valued and are generated from 
the magnitudes of the real and imaginary phase images.

By design images at coarser levels are band-limited and thus can be stored 
at lower resolution (aka downsampled).  Theoretically at least, this 
downsampling is lossless (dependent on which upsampling method used).
Some downsampling is enabled by default, since it saves on storage and
computation, but it can be disabled if desired.
If used we also keep copies of some images at the resolution
of the next finer level to make it easier to compare/correlate images across
scales.

Two pyramid builders are provided: FFT-based and convolution-based.
However the convolution-based builder is slower and may no longer be 
functional (has not been tested in a while).

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details


import torch
import collections.abc
import spyramid_filters as spf
from fft_utils import fftshift2d,ifftshift2d,freq_downsample2d, rfft_shim2d, irfft_shim2d, ifft_shim2d
from image_utils import plot_image

# This class specifies the parameters for a steerable pyramid such as which
# types of images to generate and at which levels in the pyramid
class SPyramidParams():
    # Some default values, hopefully these will only rarely be changed
    DEFAULT_ORIENTATIONS = 4
    DEFAULT_RADIAL_KERNEL = 'cos'
    DEFAULT_BOUNDARY_MODE = 'black'
    VERBOSE_PARAMS = False  # print parameter values even when they match the default values?
    
    def __init__(self,edge_levels,bandpass_levels=(0,1),*,bandpass_start=None,edge_start=None,
                 lowpass_levels=None,lowpass_start=None,orientations=DEFAULT_ORIENTATIONS,radial_kernel=DEFAULT_RADIAL_KERNEL,boundary_mode=DEFAULT_BOUNDARY_MODE):
        super().__init__()
        self.bandpass_start  :int # start level for bandpass images (note level zero corresponds to the traditional spyramid highpass image)
        self.bandpass_stop   :int # stop level for bandpass images (exclusive so no bandpass image is generated at this level)
        self.edge_start      :int # start level for (oriented) edge images
        self.edge_stop       :int # stop level for (oriented) edge images (exclusive)
        self.lowpass_start   :int # start level for lowpass images
        self.lowpass_stop    :int # stop level for lowpass images
        self.orientations    :int # Number of orientations used for edge images (typically 4)
        self.radial_kernel   :str # Radial kernel to use default is "cos", can also be "cos^2", "gauss", or "gauss0.65" (or another number)
        self.boundary_mode   :str # How to handle boundary, default is pad with zeros.  Other values are "wrap" for torus, or "wrap_x", "wrap_y" for cylinder
        if edge_levels is None: return
        # configure based on input parameters
        # levels parameters can be either the number to construct or a (start,stop) tuple indicating a specific range
        if isinstance(edge_levels,(list,tuple)):
            edge_start = edge_levels[0]
            edge_levels = edge_levels[1] - edge_levels[0]
        if isinstance(bandpass_levels,(list,tuple)):
            bandpass_start = bandpass_levels[0]
            bandpass_levels = bandpass_levels[1] - bandpass_levels[0]
        if isinstance(lowpass_levels,(list,tuple)):
            lowpass_start = lowpass_levels[0]
            lowpass_levels = lowpass_levels[1]-lowpass_levels[0]
        if bandpass_start is None and edge_start is None: bandpass_start = 0
        if bandpass_start is None: bandpass_start = edge_start - bandpass_levels
        if edge_start is None: edge_start = bandpass_start + bandpass_levels
        if lowpass_levels is None: lowpass_levels = edge_levels+1
        if lowpass_start is None: lowpass_start = edge_start
        if lowpass_levels == 0: lowpass_start = -1
        self.bandpass_start = bandpass_start
        self.bandpass_stop = bandpass_start+bandpass_levels
        self.edge_start = edge_start
        self.edge_stop = edge_start+edge_levels
        self.lowpass_start = lowpass_start
        self.lowpass_stop = lowpass_start+lowpass_levels
        self.orientations = orientations
        self.radial_kernel = radial_kernel
        self.boundary_mode = boundary_mode
#        print(f'params b{self.bandpass_start}-{self.bandpass_stop} e{self.edge_start}-{self.edge_stop} l{self.lowpass_start}-{self.lowpass_stop}')
        
    def __str__(self):
        if False:  # older format
            return f'(pyramid: bandpass {self.bandpass_range()} edge {self.edge_range()} lowpass {self.lowpass_range()} )'
        # current we're using the string (eg, 'UBBBBL:RadK=cos:Ori=4' for Freeman&Simoncelli type steerable pyramid)
        retval = ''.join(self._level_to_char(i) for i in range(self.max_stop_level())) + '_'+ str(self.max_stop_level())
        if self.orientations != self.DEFAULT_ORIENTATIONS or self.VERBOSE_PARAMS:
            retval += ':Ori='+str(self.orientations)
        if self.radial_kernel != self.DEFAULT_RADIAL_KERNEL or self.VERBOSE_PARAMS:
            retval += ':RadK='+str(self.radial_kernel)
        if self.boundary_mode != self.DEFAULT_BOUNDARY_MODE or self.VERBOSE_PARAMS:
            retval += ':Bound='+str(self.boundary_mode)
        return retval
    
    def _level_to_char(self,level:int):
        u = level >= self.bandpass_start and level < self.bandpass_stop
        e = level >= self.edge_start and level < self.edge_stop
        l = level >= self.lowpass_start and level < self.lowpass_stop
        if u:
            if e or l: raise ValueError("unsupported type")
            return 'U'
        elif e and l: return 'B'
        elif e: return 'E'
        elif l: return 'L'
        else: return 'x'
        
    # Create a pyramid parameters from a string specifying which images for build at each level 
    #  U -> unoriented (bandpass) image
    #  E -> edge (oriented bandpass) images
    #  L -> lowpass image
    #  B -> both edge and lowpass images
    #  X -> no images, skip this level
    #  <number> -> indicates total number of expected levels (very useful to catch typos and errors)
    @classmethod
    def from_str(cls,desc:str):
        # size if an optional parameter to check that the string had the expected number of levels
        size = None
        unoriented_range = (-1,-1)
        edge_range = (-1,-1)
        low_range = (-1,-1)
        def add(_range,level):
            if _range[0] == -1: return (level,level+1)
            if _range[1] != level: raise ValueError(f'levels of a type must be contiguous: {desc} {level} {_range}')
            return (_range[0],level+1)
        level = 0
        index = 0
        orientations = cls.DEFAULT_ORIENTATIONS
        radial_kernel = cls.DEFAULT_RADIAL_KERNEL
        boundary_mode = cls.DEFAULT_BOUNDARY_MODE
        fieldname = ''
        while index < len(desc):
            c = desc[index]
            index += 1
            if fieldname:
                while (index < len(desc)) and desc[index] != ':': # allow multiple character values
                    c += desc[index]
                    index += 1
                fieldname = fieldname.lower()
                if fieldname == 'ori':
                    if not c.isdigit(): raise ValueError(f'Number of orientations must be an integer: got {c} in {desc}')
                    orientations = int(c)
                elif fieldname == 'radk':
                    radial_kernel = c
                elif fieldname == 'bound':
                    boundary_mode = c
                else:
                    raise ValueError(f"Unknown fieldname {fieldname} in {desc}")
                fieldname = ''  # clear the current fieldname
            elif c.isdigit():
                while (index < len(desc)) and desc[index].isdigit(): # allow multiple digit numbers (eg '12')
                    c += desc[index]
                    index += 1
                if size is not None: raise ValueError(f'multple size specifiers in {desc}, second was {c}')
                if not c.isdigit(): raise ValueError(f'size specifier must be an integer: got {c} in {desc}')
                size = int(c)
            elif c=='U' or c=='u':
                unoriented_range = add(unoriented_range,level)
                level += 1
            elif c=='E' or c=='e': 
                edge_range = add(edge_range,level) 
                level += 1
            elif c=='L' or c=='l':
                low_range = add(low_range,level)
                level += 1
            elif c=='B' or c=='b': 
                edge_range = add(edge_range,level)
                low_range = add(low_range,level)
                level += 1
            elif c=='X' or c=='x':
                level += 1   # this level is empty
            elif c==':':
                while index<len(desc) and desc[index]!='=':
                    fieldname += desc[index]
                    index += 1
                if desc[index] != '=': raise ValueError(f"PyrParam field names must end in equals but got {fieldname} in {desc}") 
                if not fieldname: raise ValueError(f"Got empty field name in {desc}")
                index += 1
            elif c==' ' or c=='_':
                pass # skip this character 
            else:
                raise ValueError(f'Unrecognized level type "{c}" in {desc}')
        if (size is not None) and (level!=size):
            raise ValueError(f'Did not find expected number of levels in {desc}: {size} vs {level}')
        return SPyramidParams(edge_levels=edge_range,bandpass_levels=unoriented_range,lowpass_levels=low_range,orientations=orientations,radial_kernel=radial_kernel,boundary_mode=boundary_mode)                
        
    # Convert input to standard form, converting any strings to the equivalent SPyramidParams objects
    # Input can be single, a list, or a dictionary
    @classmethod 
    def normalize(cls,a):
        if a is None: return a
        if isinstance(a,SPyramidParams): return a
        if isinstance(a,str): return cls.from_str(a)  # convert string to SPyramidParam
        if isinstance(a,collections.abc.Sequence):   # convert elements in list/tuple if needed
            for i in range(len(a)):
                if isinstance(a[i],str): a[i] = cls.from_str(a[i])
        if isinstance(a,collections.Mapping):        # convert values in a dictionary if needed
            for key in a.keys():
                if isinstance(a[key],str): a[key] = cls.from_str(a[key])
        
    @classmethod
    def union(cls,a):   # Return params that cover all levels present in any of the inputs
        if isinstance(a, SPyramidParams): return a
        if isinstance(a, dict): a = a.values()
        if len(a) == 1: return a[0]
        if any(x.orientations != a[0].orientations for x in a):
            raise ValueError(f"All pyramids must use the same number of orientations: {a}")
        if any(x.radial_kernel != a[0].radial_kernel for x in a):
            raise ValueError(f"All pyramids must use the same radial kernel: {a}")
        if any(x.boundary_mode != a[0].boundary_mode for x in a):
            raise ValueError(f"All pyramids must use the same boundary mode: {a}")
        u = SPyramidParams(None)
        u.bandpass_start = min(x.bandpass_start for x in a)
        u.bandpass_stop = max(x.bandpass_stop for x in a)
        u.edge_start = min(x.edge_start for x in a)
        u.edge_stop = max(x.edge_stop for x in a)
        u.lowpass_start = min(x.lowpass_start for x in a)
        u.lowpass_stop = max(x.lowpass_stop for x in a)
        u.orientations = a[0].orientations
        u.radial_kernel = a[0].radial_kernel
        u.boundary_mode = a[0].boundary_mode
#        print(f'union {u}')
        return u
    @classmethod
    def intersection(cls,a,b):  # Return params for only the levels present in both inputs
        assert isinstance(a,SPyramidParams)
        assert isinstance(b,SPyramidParams)
        if a.orientations != b.orientations: raise ValueError(f"Pyramids must use the same number of orientations: {a} {b}")
        if a.radial_kernel != b.radial_kernel: raise ValueError(f"Pyramids must use the same radial kernel: {a} {b}")
        if a.boundary_mode != b.boundary_mode: raise ValueError(f"Pyramids must use the same boundary mode: {a} {b}")
        i = SPyramidParams(None)
        i.bandpass_start = max(a.bandpass_start,b.bandpass_start)
        i.bandpass_stop = min(a.bandpass_stop,b.bandpass_stop)
        i.edge_start = max(a.edge_start,b.edge_start)
        i.edge_stop = min(a.edge_stop,b.edge_stop)
        i.lowpass_start = max(a.lowpass_start,b.lowpass_start)
        i.lowpass_stop = min(a.lowpass_stop,b.lowpass_stop)
        i.orientations = a.orientations
        i.radial_kernel = a.radial_kernel
        i.boundary_mode = a.boundary_mode
        return i
    
    _channel_name_to_index = {'':0,'ac':0,'rg':1,'by':2}
#    _channel_index_to_name = {0:'ac', 1:'rg', 2:'by'}
    # Given a channel (can be index or name) and a set of parameters (singleton, list, or dictionary)
    # select the corresponding parameters object for this channel and return it
    @classmethod
    def select_for_channel(cls,channel,params_map):
        if isinstance(params_map,SPyramidParams): return params_map  #if there is only one params object, return it for every channel
        if isinstance(params_map,(list,tuple)):
            if isinstance(channel,str): channel = cls._channel_name_to_index[channel]
            if channel >= len(params_map): channel = len(params_map-1)
            return params_map[channel]   
        if isinstance(params_map,dict):
            if channel=='': channel = 'ac'
            #if isinstance(channel,int): channel = cls._channel_index_to_name[channel]
            return params_map[channel]
        assert False  # should never get here
        return None
        
    def bandpass_range(self): return range(self.bandpass_start,self.bandpass_stop)
    def edge_range(self): return range(self.edge_start,self.edge_stop)
    def lowpass_range(self): return range(self.lowpass_start,self.lowpass_stop)
    def crossscale_edge_range(self): return range(self.edge_start,self.edge_stop-1)
    def min_start_level(self): return min(self.bandpass_start,self.edge_start,self.lowpass_start,self.max_stop_level())
    def max_stop_level(self): return max(self.bandpass_stop,self.edge_stop,self.lowpass_stop,0)
    
""#END-CLASS------------------------------------

# Object for a "Steerable" Pyramid augmented with complex edge images, magnitude, and phase-doubled edge images
# The pyramids are created by the builders below.  They are created on-demand and thus not persistent or modules
class SPyramid():

    def __init__(self,image,params, bandpasslist,lowpasslist,edgereallist,edgeimaglist,coarser_edgereallist=None,coarser_edgeimaglist=None,*,
                 make_crossscale=True,  #Should we include the cross-scale (ie coarser) images?
                 colorname=None, temporalname=None,
                 max_reduction=1,
                 avoid_dr=True,
                 ):
        super().__init__()
        assert isinstance(params,SPyramidParams)
        # Note torch likes to store images as 4d stacks (image X channel X height X width)
        # For grayscale images the channel dimension will have length one 
        # and single images will have an image dimension of length one (eg 1x1x512x512 for single grayscale 512x512 image)
        self.image = image      # original image   
        self.cname = colorname  # The name for the channel or image type of the base image (eg, 'ac' for achromatic or luminance channel)
        self.tname = temporalname # The name for the temporal channel this pyramid belongs to (eg 't0' for the current frame)
        self._params = params   # Parameters used to define this pyramid (says which levels and components were built)
        self.bandpass = []      # band-pass images for each level (or none if not present in a level)
        self.lowpass = []       # low-pass images for each level (or none if not present in a level)
        # edge images have four channels for orientations  (ie 1x4xHeightxWidth)
        self.edge_real = []     # edge images, real part  (its convolution kernel is odd and its edge response is even)
        self.edge_imag = []     # edge images, imaginary part  (this is not currently used except to construct magn and phase doubled)
        self.edge_magn = []     # edge images, magnitude
        # images are from the next (coarser) scale (lists are one element shorter and images must match child level in size)
        # magnitude images are also included in edge_magn list but may have different resolution if downsampling is being used
        self.coarser_magn = []          # edge magnitude image from next (coarser) levels
        self.coarser_dphase_real = []   # phase doubled edge images from next levels (real part)  Can be avoided to save some memory (using imaginary parts instead)
        self.coarser_dphase_imag = []   # phase doubled edge images from next levels (imaginary part)
        self.max_reduction = max_reduction # Largest image reduction factor used in the pyramid
#        self.start_level = start_edge_level  # Starting level used for pyramid (default is zero)
#        self.start_highband_level = start_highband_level # Starting level for highbandpass images
        if avoid_dr:
            self.coarser_dphase_real = None   # not used in this case
        #---------------------------------------------------------------
        # Now that we have defined the instance variables, let's really initialize them
        #self.max_stop = max([len(bandpasslist),len(lowpasslist),len(edgeoddlist),len(edgeevenlist)])
        # Make a copy of the input lists and pad with None if needed to ensure the required matching lengths
        def _fixlist(a):
            a = list(a)
            #if len(a) < self.max_stop: a.extend([None]*(self.max_stop-len(a)))
            return a
        self.bandpass = _fixlist(bandpasslist)
        self.lowpass = _fixlist(lowpasslist)
        self.edge_real = _fixlist(edgereallist)
        self.edge_imag = _fixlist(edgeimaglist)
        # If not provided, coarser lists are just the normal lists shifted down by one level (only works if there is no downsampling between levels)
        if coarser_edgereallist is None: 
            coarser_edgereallist = edgereallist[1:] + [None]
            if params.edge_start>0: coarser_edgereallist[params.edge_start-1]=None
        else:
            coarser_edgereallist = _fixlist(coarser_edgereallist)
        if coarser_edgeimaglist is None: 
            coarser_edgeimaglist = edgeimaglist[1:] + [None]
            if params.edge_start>0: coarser_edgeimaglist[params.edge_start-1]=None
        else:
            coarser_edgeimaglist = _fixlist(coarser_edgeimaglist)
        #----------------------------------------------------------------
        # Generate derived edge images
        for (re,im) in zip(self.edge_real,self.edge_imag):
            mag = None
            if re is not None:
                # We add a tiny epsilon to ensure argument to sqrt() is >0,  (sqrt(0) produces NaNs in its gradient calculatons and if used as a denominator) 
                mag = torch.sqrt(re*re + im*im + torch.finfo(re.dtype).tiny)
            self.edge_magn.append(mag)
        if make_crossscale:
            # construct phase-doubled images from next coarser scales edge-filtered images
            # note: phase-doubling swaps the even/odd-ness of the edge filter (because our filters are defined to be imaginary-valued)
            for (re,im) in zip(coarser_edgereallist,coarser_edgeimaglist):
                if re is not None:
                    # We add a tiny epsilon to ensure argument to sqrt() is >0,  (sqrt(0) produces NaNs in its gradient calculatons and if used as a denominator) 
                    mag = torch.sqrt(re*re + im*im + torch.finfo(re.dtype).tiny)
                    self.coarser_magn.append(mag)
                    if not avoid_dr:
                        self.coarser_dphase_real.append( (re*re-im*im)/mag )
                    self.coarser_dphase_imag.append( (2*re*im)/mag )
                else:
                    self.coarser_magn.append(None)
                    if not avoid_dr:
                        self.coarser_dphase_real.append(None)
                    self.coarser_dphase_imag.append(None)
        def _check_range(l,r):
            for i,x in enumerate(l):
                assert (x is not None) if i in r else (x is None)
        _check_range(self.bandpass,self._params.bandpass_range())
        _check_range(self.lowpass,self._params.lowpass_range())
        _check_range(self.edge_real,self._params.edge_range())
        _check_range(self.edge_magn,self._params.edge_range())
        if make_crossscale:
            _check_range(self.coarser_magn,self._params.crossscale_edge_range())

    def original_image(self):
        return self.image
        
    def high_pass_image(self):
        return self.bandpass[0]

    def band_pass_image(self,level):
        return self.bandpass[level]
        
    def low_pass_image(self,level):
        return self.lowpass[level]
    
    def edge_real_images(self,level): #real-edge-component has an *odd* convolution kernel and the corresponding edge response in images is *even*
        return self.edge_real[level]
    
    def edge_imag_images(self,level): #imaginary-edge-component has an *even* convolution kernel and the corresponding edge response in images is *odd*
        return self.edge_imag[level]
    
    def edge_magnitude_images(self,level):
        return self.edge_magn[level]
    
    def coarser_magnitude_images(self,level):
        return self.coarser_magn[level]
    
    def dphase_real_images(self,level):   #note phase doubling flips the even/odd-ness of the image properties
        if self.coarser_dphase_real is None: return None
        return self.coarser_dphase_real[level]
    
    def dphase_imag_images(self,level):   #note phase doubling flips the even/odd-ness of the image properties
        return self.coarser_dphase_imag[level]
    
    def params(self): return self._params
    
    def plot_component_images(self):
        plot_image(self.original_image(),title='original image')
        for i in self.params().bandpass_range():
            plot_image(self.band_pass_image(i),title=f'band pass {i}' if i>0 else 'high pass 0')
        for i in self.params().lowpass_range():
            plot_image(self.low_pass_image(i),title=f'low pass {i}')
        for i in self.params().edge_range():
            er = self.edge_real_images(i)
            ei = self.edge_imag_images(i)
            em = self.edge_magnitude_images(i)
            for ori in range(er.size(1)):
                plot_image(torch.cat((er[0,ori,:,:],ei[0,ori,:,:],em[0,ori,:,:]),-1), title=f'edges {i},{ori} (real,imag,magn)')
        for i in self.params().crossscale_edge_range():
            er = self.edge_real_images(i)
            dr = self.dphase_real_images(i)
            di = self.dphase_imag_images(i)
            for ori in range(er.size(1)):
                if dr is None: 
                    plot_image(torch.cat((er[0,ori,:,:],di[0,ori,:,:]),-1), title=f'phase {i},{ori} (real,dphase_imag)')
                else:
                    plot_image(torch.cat((er[0,ori,:,:],di[0,ori,:,:],dr[0,ori,:,:]),-1), title=f'phase {i},{ori} (real,dphase_imag,dphase_real)')
""#END-CLASS------------------------------------

# Object that can build steerable pyramids for images using convolutional filters
# Note: Has not been used in a while and may not be up to date
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! We reccomend that you use SPyramidFourierBuilder instead !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class SPyramidConvolutionalBuilder_experimental(torch.nn.Module):
    
    def __init__(self,params):
        super().__init__()
        self.filters_bandpass = None    # H0 high pass filter
        self.filters_lowpass = None    # list of low pass filters, finest to coarsest
        self.filters_edge_real = None  # list of stacks of edge filters for each level (real part)
        self.filters_edge_imag = None  # (imaginary part)
        self._build_conv_filters(SPyramidParams.union(params))
        self.params_map = params
        
    def _build_conv_filters(self,uparams):
        # Create the initial high and low pass filters in fourier domain and convert to spatial domain
        # We store the filters as parameters in parameterlist so module superclass can move them to GPU etc.
        minstart = uparams.min_start_level()
        maxstop = uparams.max_stop_level()
        self.filters_lowpass = torch.nn.ParameterList([None]*maxstop)  
        self.filters_edge_real = torch.nn.ParameterList([None]*maxstop)
        self.filters_edge_imag = torch.nn.ParameterList([None]*maxstop)
        self.filters_bandpass = torch.nn.ParameterList([None]*maxstop)
        def _wrap(filt):
            return torch.nn.Parameter(filt,requires_grad=False)    
        
        # create the low-pass and high-pass filters that we will need to build the other filters
        hi_f = [None]*maxstop
        lo_f = [None]*maxstop
        for i in range(minstart,maxstop):
            [hi_f[i], lo_f[i]] = spf.create_fourier_high_low_filters(max_freq=0.5**i)
        # create the band-pass filters for required levels (note: level zero is a special case since it is limited by image resolution rather than a low-pass filter)
        for i in uparams.bandpass_range():
            prev_lo_f = lo_f[i-1] if i>0 else 1
            self.filters_bandpass[i] = _wrap(self._fourier_to_spatial_filter_real(prev_lo_f*hi_f[i]))
        # create the low-pass filters for required levels
        for i in uparams.lowpass_range():
            prev_lo_f = lo_f[i-1] if i>0 else 1
            self.filters_lowpass[i] = _wrap(self._fourier_to_spatial_filter_real(prev_lo_f))
        # Create azimuthal or oriented filters used in constructing edge filters
        azimuthal_f = spf.create_fourier_oriented4_imaginary_filters()
        # Now construct edge filters for each requested level 
        for i in uparams.edge_range():
            prev_lo_f = lo_f[i-1] if i>0 else 1
            [edges_real, edges_imag] = self._fourier_imaginary_to_spatial_filters(prev_lo_f*hi_f[i]*azimuthal_f)
            self.filters_edge_real[i] = _wrap(edges_real)
            self.filters_edge_imag[i] = _wrap(edges_imag)

            
    @staticmethod
    def _fourier_to_spatial_filter_real(fourier):
        # Convert real fourier filter to convoluational filter in spatial domain
        conv = spf.fourier_real_to_real_convolutional_filter(fourier)
        # Trim filter to remove regions where it is nearly zero anyway (smaller filters are cheaper)
        conv = spf.trim_convolutional_filter(conv)
        # And add single channel dimension 
        return conv.unsqueeze(0)
    
    @staticmethod
    def _fourier_imaginary_to_spatial_filters(fourier):
        comp = spf.fourier_imaginary_to_convolutional_filter(fourier)
        #print(comp.size())
        comp = spf.trim_convolutional_filter(comp,dim=(1,2))
        conv_real = comp[:,:,:,0]
        conv_imag = comp[:,:,:,1]
        return (conv_real.unsqueeze(1),conv_imag.unsqueeze(1))
         
    def build_spyramid(self, image, *, name='', params=None, make_crossscale=True):
        if params is None:
            params = SPyramidParams.select_for_channel(name, self.params_map)
        maxstop = params.max_stop_level()
        # Build requested bandpass images
        bandlist = [None]*maxstop
        for i in params.bandpass_range():
            bandlist[i] = spf.conv2d_keep_size(image,self.filters_bandpass[i])
        # Build requested lowpass images
        lowlist = [None]*maxstop
        for i in params.lowpass_range():
            lowlist[i] = spf.conv2d_keep_size(image,self.filters_lowpass[i])
        # Build requested edge images
        edgereal = [None]*maxstop
        edgeimag = [None]*maxstop
        for i in params.edge_range():
            edgereal[i] = spf.conv2d_keep_size(image,self.filters_edge_real[i])
            edgeimag[i] = spf.conv2d_keep_size(image,self.filters_edge_imag[i])
        return SPyramid(image,params,bandlist,lowlist,edgereal,edgeimag,name=name)
        
    def high_pass_filter(self):
        return self.filter_bandpass[0]
    
    def band_pass_filters(self):
        return self.filters_bandpass
    
    def low_pass_filters(self):
        return self.filters_lowpass
    
    def edge_real_filters(self):
        return self.filters_edge_real
    
    def edge_imag_filters(self):
        return self.filters_edge_imag
    
""#END-CLASS------------------------------------------------------

# Object that can build steerable pyramids for images using FFT transforms and fourier-space filters  
# The boundary of the images can be expanded with zeroes to simulate the result for a non-repeating image  
# Component images that have been low-pass filtered can also optionally be downsampled (to save memory and computation)
# and the maximum amount of downsampling can also be limited (eg, to avoid neeing fractional pixel coordinates)   
class SPyramidFourierBuilder(torch.nn.Module):
    
    # This method needs to know the size of the images so it can prebuild its fourier filters
    def __init__(self,image_size,params,
                 downsample=True,max_downsample_factor=64):
        super().__init__()
        params = SPyramidParams.normalize(params)
        unionparams = SPyramidParams.union(params) # Take union of pyramid parameters so we can construct according to any of the specified sets of parameters
        self.filters_bandpass = None    # list of band-pass filters (first is the traditional high pass filter)
        self.filters_lowpass = None     # list of low-pass filters, finest to coarsest
        self.filters_edge = None        # list of stacks of edge filters for each level (real part)
        if not downsample: max_downsample_factor = 1     # If not downsampling, then max reduction factor is one
        self.max_downsample_factor = max_downsample_factor
        self.pad_mode = 'constant'
        self.padX = self.padY = 2 ** (unionparams.max_stop_level()-1) # Add zero padding around image to reduce wraparound effects of FFT
        boundary = unionparams.boundary_mode
        if boundary == 'black' or boundary == 'zeros':
            pass
        elif boundary == 'wrap' or boundary == 'circular':
            self.padX = self.padY = 0   # wraparound (aka circular or torus) is default for fourier method so no padding needed in this case
        elif boundary == 'wrap_x' or boundary == 'circular_x':
            self.padX = 0           # Wrap around but only in the x direction
        elif boundary == 'wrap_y' or boundary == 'circular_y': 
            self.padY = 0           # Wrap around but only in the y direction
        else: raise ValueError(f'Unrecognized boundary mode: {boundary}')
#        elif isinstance(pad_boundary,str): self.pad_mode = pad_boundary
#        else: self.padX = self.padY = pad_boundary
        self._build_fourier_filters(image_size,unionparams)
        self.params_map = params
            
    # Build fourier space filters (must match size of actual image they will be applied to)
    def _build_fourier_filters(self,image_size,uparams):
        if isinstance(image_size,torch.Tensor): image_size = image_size.size()  # If input was an image, convert it to a size
        if len(image_size) > 2:
            image_size = (image_size[-2],image_size[-1])   #Keep only last two dimensions (height,width) as size
        # Add zero padding of the boundaries if needed
        image_size = (image_size[0]+2*self.padY, image_size[1]+2*self.padX)
        # We store the filters as parameters in parameterlist so module superclass can move them to GPU etc.
        def _wrap(filt):
            return torch.nn.Parameter(filt,requires_grad=False)
        minstart = uparams.min_start_level()
        maxstop = uparams.max_stop_level()
        self.filters_lowpass = torch.nn.ParameterList([None]*maxstop)  # list of low-pass filters, finest to coarsest
        self.filters_edge = torch.nn.ParameterList([None]*maxstop)     # list of stacks of edge filters for each level (real part)
        self.filters_bandpass = torch.nn.ParameterList([None]*maxstop) # list of band-pass filters
        
        # create the low-pass and high-pass filters that we will need to build the other filters
        hi_f = [None]*maxstop
        lo_f = [None]*maxstop
        if uparams.radial_kernel == 'cos':
            self.downsample_kernel_bias = -1  #kernel band-limited after one level
#            print(f'pyrbuild  {(minstart,maxstop)}')
            for i in range(minstart,maxstop):
                [hi_f[i], lo_f[i]] = spf.create_fourier_high_low_filters(max_freq=0.5**i,size=image_size)
        elif uparams.radial_kernel == 'cos^2':
            self.downsample_kernel_bias = -1
            for i in range(minstart,maxstop):
                [hi_f[i], lo_f[i]] = spf.create_fourier_high_low_filters(max_freq=0.5**i,size=image_size)
                lo_f[i] = lo_f[i]**2
                hi_f[i] = hi_f[i]**2
        elif uparams.radial_kernel.startswith('gauss'):
            sigma = 0.65
            if len(uparams.radial_kernel) > 5:
                sigma = float(uparams.radial_kernel[5:])
#                print(f'gauss sigma is {sigma}')
                if sigma <= 0 or sigma > 1: raise ValueError(f'invalid sigma for radial gaussian kernel {sigma}')
            self.downsample_kernel_bias = -2   #band-limited after two levels
            [_,lo_cos0] = spf.create_fourier_high_low_filters(max_freq=1,size=image_size)
            for i in range(minstart,maxstop):
                [hi_f[i], lo_f[i]] = spf.create_fourier_gaussian_high_low_filters(freq=0.5**i,size=image_size,sigma=sigma)
                lo_f[i] = torch.min(lo_f[i],lo_cos0) # force kernel to go to zero  always for highest image frequencies
        else:
            raise ValueError(f'Unknown radial kernel type {uparams.radial_kernel}')
        # create the band-pass filters for required levels (note: level zero is a special case since it is limited by image resolution rather than a low-pass filter)
        for i in uparams.bandpass_range():
            prev_lo_f = lo_f[i-1] if i>0 else 1
            self.filters_bandpass[i] = _wrap(self._convert_real_filter(prev_lo_f*hi_f[i]))
#            plot_image(self.filters_bandpass[i][0,:,:,:,0],f'bandpass filter {i}')
        # create the low-pass filters for required levels
        for i in uparams.lowpass_range():
            prev_lo_f = lo_f[i-1] if i>0 else 1
            self.filters_lowpass[i] = _wrap(self._convert_real_filter(prev_lo_f))
        # create oriented edge filters for required levels
        # Create azimuthal or oriented filters used in constructing edge filters        
        azimuthal_f = spf.create_fourier_oriented_imaginary_filters(size=image_size,orientations=uparams.orientations)
#        azimuthal_f = spf.create_fourier_oriented4_imaginary_filters(size=image_size)
        # note: we store these as real filters (effectively multiplying by i), which swaps the real and imaginary components (and negates one)
        #  since we need both components we swap the two edge images back during build stage (and the negation doesn't matter for us)
        for i in uparams.edge_range():
            prev_lo_f = lo_f[i-1] if i>0 else 1
            self.filters_edge[i] = _wrap(self._convert_stacked_filters(prev_lo_f*hi_f[i]*azimuthal_f))
#            plot_image(self._convert_real_filter(prev_lo_f*hi_f[i])[0,:,:,:,0],f'bandpass filter {i}')
#            plot_image(self.filters_edge[i][0,0,:,:,0],f'edge filter {i}')
#            plot_image(self._convert_stacked_filters(azimuthal_f)[0,0,:,:,0],f'azimuthal filter')
            
    @staticmethod
    def _convert_real_filter(fourier):
        # Shift filter into defualt fft format and add dimensions for image, channel, and complex-component
        return ifftshift2d(fourier)[None,None,:,:,None]
    
    @staticmethod
    def _convert_stacked_filters(fourier):
        # Shift filter into defualt fft format and add dimensions for image, channel, and complex-component
        return ifftshift2d(fourier)[None,:,:,:,None]
            
    # Build a steerable pyramid for the given image
    #  params - Specifies which image levels and types to build in the pyramid
    #  make_crossscale - Should we generate the images used for cross-scale correlations (phase-double and reduction matched images for neighboring scales)
    def build_spyramid(self, image, *, colorname='', temporalname='', params=None, make_crossscale=True):
        if params is None:
            params = SPyramidParams.select_for_channel(colorname, self.params_map)
        orig_image = image
        # Pad image boundary with zeros (if needed)
        padX = self.padX
        padY = self.padY
        if (padX != 0) or (padY != 0):   # Pad image by adding extra elements around the boundary
            image = torch.nn.functional.pad(image,(padX,padX,padY,padY),mode=self.pad_mode)  
        def unpad(image,reduction=1):            # Function to removed padding after filering and IFFT
            #return image
            if (padX != 0) or (padY != 0):
                prX = padX//reduction
                prY = padY//reduction
                return torch.nn.functional.pad(image,(-prX,-prX,-prY,-prY))
            else:
                return image
        # Used for selecting degree of image downsampling (when enabled)
        def reduction_factor(level):
            return min(2**max(level+self.downsample_kernel_bias,0),self.max_downsample_factor)
        # Compute FFT of image
#        freq_img = torch.rfft(image,2,onesided=False)
        freq_img = rfft_shim2d(image)  #Use shim for new FFT API
        # Apply various filters by multiplication and then use inverse FFT to get results
        maxstop = params.max_stop_level()
        # Build requested bandpass images
        bandlist = [None]*maxstop
        for i in params.bandpass_range():
            freq_bandp = freq_img*self.filters_bandpass[i]
            reduction = reduction_factor(i)
            if reduction > 1: freq_bandp = freq_downsample2d(freq_bandp,reduction)/(reduction**2)
            bandlist[i] = unpad(irfft_shim2d(freq_bandp),reduction)
            del freq_bandp
        # Build requested lowpass images
        lowlist = [None]*maxstop
        for i in params.lowpass_range():
            freq_lowp = freq_img*self.filters_lowpass[i]
            reduction = reduction_factor(i)
            # Optionally reduce image size by removing high frequencies (and compensating for reduced size of image)
            if reduction > 1: freq_lowp = freq_downsample2d(freq_lowp,reduction)/(reduction**2)
            lowlist[i] = unpad(irfft_shim2d(freq_lowp),reduction)
            del freq_lowp  #allow freq_lowp to be garbage collected here
        # Build requested oriented edge images
        edgereal = [None]*maxstop          # List of edge images for each scale (real part)
        edgeimag = [None]*maxstop          # List of edge images for each scale (imaginary part)
        edgereal_ncs = [None]*maxstop      # List of next-coarser-scale edge images (real part) used for cross-scale correlations
        edgeimag_ncs = [None]*maxstop      # List of next-coarser-scale edge images (imaginary part)
        prev_reduction = None
        max_reduction = 1
        for i in params.edge_range():
            freq_edge = freq_img*self.filters_edge[i]
            reduction = reduction_factor(i)
            if reduction > 1:
                edge_c = ifft_shim2d(freq_downsample2d(freq_edge,reduction)/(reduction**2))
            else:
                edge_c = ifft_shim2d(freq_edge)
            edgereal[i] = unpad(edge_c[:,:,:,:,1],reduction) #note swap back of real and imaginary components here
            edgeimag[i] = unpad(edge_c[:,:,:,:,0],reduction)
            del edge_c
            # If the finer scale edge level exists and we want cross-scale correlations, then store a version of this edge image matching the finer scales resolution and index
            if (i>0) and (edgereal[i-1] is not None) and make_crossscale:  # are the next higher frequency edge images present?
                prev_reduction = reduction_factor(i-1)
                if reduction != prev_reduction:     
                    # If previous level used a different reduction factor, then create a matching version (for cross-scale correlations)
                    #freq_parent = freq_downsample2d(freq_edge,prev_reduction)/(prev_reduction**2)
                    edge_parent = ifft_shim2d(freq_downsample2d(freq_edge,prev_reduction)/(prev_reduction**2))
                    edgereal_ncs[i-1] = unpad(edge_parent[:,:,:,:,1],prev_reduction)
                    edgeimag_ncs[i-1] = unpad(edge_parent[:,:,:,:,0],prev_reduction)
                    del edge_parent
                else:
                    # If image resolution did not change, then just reuse the already created edge images
                    edgereal_ncs[i-1] = edgereal[i] 
                    edgeimag_ncs[i-1] = edgeimag[i]
            del freq_edge  # allow image to be garbage collected here
        return SPyramid(orig_image,params,bandlist,lowlist,edgereal,edgeimag,edgereal_ncs,edgeimag_ncs,colorname=colorname,temporalname=temporalname,make_crossscale=make_crossscale,max_reduction=max_reduction)

    def high_pass_filter(self):
        return self.filters_highbandpass[0]
#        return self.filter_highpass
    
    def band_pass_filters(self):
        return self.filters_bandpass
    
    def low_pass_filters(self):
        return self.filters_lowpass
    
    def edge_filters(self):
        return self.filters_edge
    
""#END-CLASS------------------------------------

# Low pass filter an image in approximately the same way it would be done in 
# a steerable pyramid at the given level.  Does not construct other pyramid components
# Currently this does not use the GPU and has not been optimized for efficiency
def lowpass_filter_image(image,level,preserve_edge_brightness=False):
    if level < 0: raise ValueError(f"level {level} must be >0")
    pad = 2 ** (level)
    pad_img = torch.nn.functional.pad(image,(pad,)*4)  # Pad image by adding zeroes around the boundary
    image_size = (pad_img.size(-2),pad_img.size(-1))
    # Compute FFT of image
    freq_img = rfft_shim2d(pad_img)
    [hi_f,lo_f] = spf.create_fourier_high_low_filters( max_freq=(0.5**level),size=image_size )
    #plot_image(lo_f)
    freq_lowp = freq_img * (ifftshift2d(lo_f)[None,None,:,:,None])
    lo_img = irfft_shim2d(freq_lowp)
    #plot_image(lo_img)
    res_img = torch.nn.functional.pad(lo_img,(-pad,)*4)
    if preserve_edge_brightness:
        filternorm = lowpass_filter_image(torch.ones_like(image),level,False)
        res_img = res_img / filternorm
    return res_img


def test_fourier():
    img = torch.zeros(1,1,128,128)
    img[0,0,31,31] = 1
    #img = load_image_gray('cat256.png')
    params = SPyramidParams(4)
    builderC = SPyramidConvolutionalBuilder_experimental(params)
    spyrC = builderC.build_spyramid(img)
    builder = SPyramidFourierBuilder(img,pad_boundary=True,downsample=False)
    #print(builder.high_pass_filter().size())
    #plot_image(builder.high_pass_filter())
    #plot_image(builder.low_pass_filters()[0])
    #print(builder.edge_filters()[0].size())
    #plot_image(builder.edge_filters()[0][0,:,:])
    spyr = builder.build_spyramid(img)
    #plot_image(torch.cat( (spyrC.low_pass_images()[-1],spyr.low_pass_images()[-1]), -1))
    ell = 3+1
    ori = 2
    if ell==0: plot_image(torch.cat( (spyrC.high_pass_image(),spyr.high_pass_image()), -1))
    plot_image(torch.cat( (spyrC.low_pass_image(ell),spyr.low_pass_image(ell)), -1))
    plot_image(torch.cat( (spyrC.edge_real_images(ell)[0,ori,:,:],spyr.edge_real_images(ell)[0,ori,:,:]), -1))
    plot_image(spyrC.edge_real_images(ell)[0,ori,:,:] - spyr.edge_real_images(ell)[0,ori,:,:])
    plot_image(torch.cat( (spyrC.edge_imag_images(ell)[0,ori,:,:],spyr.edge_imag_images(ell)[0,ori,:,:]), -1))
    #plot_image(spyr.low_pass_images()[1])
    
def test_downsample():
    img = load_image_gray('../sampleimages/stripespot256.png')
    img = torch.ones(1,1,128,128)
    def relu(x):
        if x > 0: return x
        return 0
    level_offset = 1
    params1 = SPyramidParams(6,edge_start=relu(level_offset)+1)
    params2 = SPyramidParams(6,edge_start=relu(-level_offset)+1)
    builder = SPyramidFourierBuilder(img,params1,downsample=True)
    builder2 = SPyramidFourierBuilder(img,params2,downsample=False)
    spyr = builder.build_spyramid(img)
    spyr2 = builder2.build_spyramid(img)
    ell = 3
    plot_image(spyr.high_pass_image(),center_zero=False,title='high pass')
    plot_image(spyr2.high_pass_image(),center_zero=False,title='high pass')
    plot_image(spyr.low_pass_image(ell),center_zero=False)
    plot_image(spyr2.low_pass_image(ell),center_zero=False)
    #plot_images(torch.unbind(spyr.edge_real_images()[ell].squeeze(),0))
    #plot_images(torch.unbind(spyr2.edge_real_images()[ell].squeeze(),0))
    plot_images(torch.unbind(spyr.edge_imag_images(ell).squeeze(),0),num_rows=1)
    plot_images(torch.unbind(spyr2.edge_imag_images(ell).squeeze(),0),num_rows=1)
    
def test_circular():
    wh = torch.ones(1,1,32,32)
    bl = torch.zeros(1,1,32,32)
    img = torch.cat([torch.cat([wh,wh], dim=-1),torch.cat([wh,bl],dim=-1)],dim=-2)
    plot_image(img,title='image')
    params = SPyramidParams(2)
    for pm in ['black','wrap','wrap_x','wrap_y']:
        params.boundary_mode = pm
        builder = SPyramidFourierBuilder(img,params)
        spyr = builder.build_spyramid(img)
        plot_image(spyr.low_pass_image(2),title=pm)
    

def make_examples():
    # type %matplotlib auto   into console to get plots in popup window instead
    #img = torch.zeros(1,1,65,65); img[0,0,32,32]=65*65
    #img = load_image_gray('sampletextures/simplesmall/stripe256.png')
    #img = load_image_gray('sampletextures/simplesmall/spot256.png')
    img = load_image_gray('../sampleimages/stripespot256.png')
    params = SPyramidParams(4,boundary_mode='wrap')
    builder = SPyramidFourierBuilder(img,params,downsample=False)
    spyr = builder.build_spyramid(img)
    #plot_images(spyr.low_pass_images()[1:],center_zero=False)
    plot_image(spyr.low_pass_image(4),center_zero=False)
    #plot_image(torch.cat( spyr.low_pass_images(), -1))
    ell = 2
    ori = 0
    plot_image(spyr.low_pass_image(ell),center_zero=True)
    #plot_images( [[spyr.original_image()], [spyr.low_pass_images()[ell]]] , center_zero=False)
    #plot_images(torch.unbind(spyr.edge_real_images()[ell].squeeze(),0))
    #plot_images(torch.unbind(spyr.edge_imag_images()[ell].squeeze(),0))
    #plot_images( [torch.unbind(spyr.edge_real_images()[ell].squeeze(),0), torch.unbind(spyr.edge_imag_images()[ell].squeeze(),0) ] )
    #plot_image(torch.cat( (spyr.edge_real_images()[ell][0,ori,:,:],), -1))
    #plot_images([spyr.edge_real_images()[ell][0,ori,:,:],spyr.edge_imag_images()[ell][0,ori,:,:],spyr.edge_magnitude_images()[ell][0,ori,:,:]])
    #plot_images([spyr.edge_real_images()[ell][0,ori,:,:],spyr.edge_real_images()[ell+1][0,ori,:,:],spyr.dphase_real_images()[ell][0,ori,:,:]])
    
    plot_image(spyr.image,center_zero=False,colorbar=False)
    plot_image(spyr.low_pass_image(4),center_zero=False,colorbar=False)
    
    plot_image(spyr.high_pass_image(),colorbar=False)
    
    plot_image(spyr.edge_magnitude_images(2)[0,0,:,:],colorbar=False)
    plot_image(spyr.edge_magnitude_images(3)[0,1,:,:],colorbar=False)
    
    plot_image(spyr.edge_real_images(2)[0,0,:,:],colorbar=False,title='Edge real')
    plot_image(spyr.edge_imag_images(2)[0,0,:,:],colorbar=False,title='Edge imaginary')
    plot_image(spyr.edge_real_images(3)[0,1,:,:],colorbar=False)
    spyr.plot_component_images()
    
def plot_convolution_kernels():

    siz = 128
    img = torch.zeros(1,1,siz,siz)
    img[0,0,siz//2,siz//2] = 1
    levels = 6
    #img = load_image_gray('cat256.png')
    params = SPyramidParams(levels)
    builder = SPyramidFourierBuilder(img,params,downsample=False)
    spyr = builder.build_spyramid(img)
    klist = []
    for i in spyr.params().lowpass_range():
        klist.append(spyr.low_pass_image(i)*(4**(i-1)))
    plot_image(torch.cat(klist,-1),title=f'low pass kernels for {levels} levels')
    klist = []
    ori = 0
    for i in spyr.params().edge_range():
        klist.append(spyr.edge_real_images(i)[0,ori,:,:]*(4**(i-1)))
    plot_image(torch.cat(klist,-1),title=f'edge kernels (real) for {levels} levels')
#    spyr.plot_component_images()
#    plot_image(torch.cat( (spyrC.edge_real_images()[ell][0,ori,:,:],spyr.edge_real_images()[ell][0,ori,:,:]), -1))

def test_minibatch():
    siz = 128
#    img = torch.ones(2,1,siz,siz)
    levels = 6
    params = SPyramidParams(levels)
    catimg = load_image_gray('../sampleimages/cat256.png')
    img = torch.cat([catimg,torch.ones_like(catimg)],0)
    builder = SPyramidFourierBuilder(img,params,downsample=True)
    spyr = builder.build_spyramid(img)
    plot_images(torch.unbind(spyr.low_pass_image(3)))
    ori = 0
    plot_images(torch.unbind(spyr.edge_magnitude_images(3)[:,ori:ori+1,:,:]))
    
def test_plot_fourier_filters():
    siz = 64
    elevels = 4
    hblevels = 1
    img = torch.zeros(1,1,siz,siz)
    params = SPyramidParams(elevels,edge_start=hblevels)
    builder = SPyramidFourierBuilder(img,params,downsample=False)
    for i,filt in enumerate(builder.band_pass_filters()):
        if filt is None: continue
        plot_image(filt.squeeze(-1),title=f'band-pass filter {i}')
    for i,filt in enumerate(builder.low_pass_filters()):
        if filt is None: continue
        plot_image(filt.squeeze(-1),title=f'low-pass filter {i}')
    for i,filts in enumerate(builder.edge_filters()):
        if filts is None: continue
        plot_images(filts.squeeze(-1).unbind(1),title=f'edge filters {i}')   
        
def plot_candidate_kernels():
    def _convert_stacked_filters(fourier):
        # Shift filter into defualt fft format and add dimensions for image, channel, and complex-component
        return ifftshift2d(fourier)[None,:,:,:,None]
    def _convert_kern_real(fourier):
        # Convert fourier kernel to spatial and return real and imag parts
        kern = fftshift2d(torch.ifft(fourier,2))
        return kern[:,:,:,:,0],kern[:,:,:,:,1]

    siz = 128
    img = torch.zeros(1,1,siz,siz)
    img[0,0,siz//2,siz//2] = 1
    orientations = 4
    ori = 0
    
    start = 3
    stop = start+1
    [_,lo_f] = spf.create_fourier_high_low_filters(max_freq=0.5**start,size=siz)
    [hi_f,_] = spf.create_fourier_high_low_filters(max_freq=0.5**stop,size=siz)
    lo_f = lo_f**2
    hi_f = hi_f**2
#    azimuthal_f = spf.create_fourier_oriented4_imaginary_filters(size=siz)
    azimuthal_f = spf.create_fourier_oriented_even_imaginary_filters(size=siz,orientations=orientations)
#    azimuthal_f = torch.ones_like(azimuthal_f)
    edge_f = _convert_stacked_filters(lo_f*hi_f*azimuthal_f)
    edge_f = torch.cat([torch.zeros(1,1,siz,siz,1),edge_f.narrow(1,ori,1)],dim=-1)
   
#    plot_image(_convert_kern_real(edge_f)[0],title=f'{orientations} orientations real edge kernel',savefile=f'edge{orientations}_real_kernel.png')
#    plot_image(_convert_kern_real(edge_f)[0],title=f'{orientations} orientations real edge kernel')

    unoriented_f = torch.cat([ifftshift2d(lo_f*hi_f)[None,None,:,:,None],torch.zeros(1,1,siz,siz,1)],dim=-1)
    unoriented_img = fftshift2d(irfft_shim2d(unoriented_f))
    plot_image(unoriented_img,title=f'Bandpass unoriented kernel',savefile=f'bandpass_unoriented_kernel.png',)
    
if __name__ == "__main__":   # execute main() only if run as a script
    from image_utils import load_image_gray, plot_image, plot_images, plot_image_channels, plot_image_stack
#    test_fourier()
#    test_downsample()
#    test_circular()
#    make_examples()        
#    plot_convolution_kernels()
#    test_minibatch()
#    test_plot_fourier_filters()
    plot_candidate_kernels()
