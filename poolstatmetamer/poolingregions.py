#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:06:15 2019

 Code to pool (ie average) values from an image over specified windows/kernels 
 and return the result as a image (usually smaller than the input).

 Regions can be the whole image or fixed-size windows (which can be overlapping)
 Used to reduce image statistics to regional averages when computing metamers

 A pooling object must implement the pool_stat() method to perform its pooling
 Its image argument may have already been reduced/downsampled in some cases
 The original_width argument can be used to detect this and size the pooling kernel appropriately

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details



import torch
import torch.nn.functional as F
import math
from fractions import Fraction
import collections.abc
import ast
import imageblends as blend
import imagefilters as filters
from typing import NamedTuple, Optional
from image_utils import plot_image

# This class specifies the parameters for pooling regions such as width, kernel type, etc.
# Can be used to generate corresponding pooling kernels
class PoolingParams():
    DEFAULT_KERNEL = 'Trig'
    DEFAULT_STRIDE = 1/4
    DEFAULT_MESA_FRACTION = 1/2
    DEFAULT_BOUNDARY_MODE = 'zeros'
    DEFAULT_BOUNDARY_REGIONS = -1     # number of pooling regions extending beyond image borders, -1 means select automatically
    DEFAULT_FORCED_PAD = None          # if set, overrides boundary regions to used a fixed padding size around the edges of the image for pooling
    DEFAULT_CIRCULAR = False
    DEFAULT_SHRINK = 1
    VERBOSE_PARAMS = False  # print parameter values even when they match the default values?
    
    def __init__(self,width,kernel=DEFAULT_KERNEL,stride_fraction=DEFAULT_STRIDE,*,mesa_fraction=DEFAULT_MESA_FRACTION,boundary_regions=DEFAULT_BOUNDARY_REGIONS,boundary_mode=DEFAULT_BOUNDARY_MODE,forced_pad=DEFAULT_FORCED_PAD,
                 circular=DEFAULT_CIRCULAR,shrink=DEFAULT_SHRINK,mask=None,pixel_offset=None,force_unit_stride=False):
        #TODO: extend to support multi-width gaze-centric pooling
        super().__init__()
        self.width           :Optional[int]   # width the pooling kernel (width of its support which is assumed to be square)
        self.kernel          :str   # type of the kernel (might be extended later ot allow supplying a custom kernel tensor)
        self.stride_fraction :float # distance between neighboring kernels as a fraction of their width (controls degree of overlap)
        self.mesa_fraction   :float # fraction of 1d slide through kernel where the kernel has its maximum value (ie size of kernel's flat top region)
        self.boundary_regions:int   # Number of boundary region extending beyond image boundary (-1 means use a default value) 
        self.boundary_fixed_pad:Optional[int] # If set, overrides boundary_region setting to provide a fixed padding around the image for pooling
        self.boundary_mode   :str   # Padding type to be used for image when pooling regions extend beyond its boundary (zeros, wrap, wrap_x, or wrap_y)
        self.circular        :bool  # Should kernel be circularlly symmetric? (default is separable kernels that are not circular)
        self.mask            :Optional[torch.Tensor]
        self.pixel_offset    :Optional[list[int]] # Optional offset to kernel locations
        self.force_unit_stride:bool # Special mode for visualizations and debugging, force stride==1 at all resolutions
        # configure based on input parameters, current just copy them but will become more adaptive as we add more features
        self.width = width
        self.kernel = kernel
        self.stride_fraction = stride_fraction
        self.mesa_fraction = mesa_fraction
        self.boundary_regions = boundary_regions
        self.boundary_forced_pad = forced_pad
        self.boundary_mode = boundary_mode.lower()
        self.circular = circular
        self.shrink = shrink
        self.mask = mask
        self.pixel_offset = pixel_offset
        self.force_unit_stride = force_unit_stride
        if mask is not None: raise ValueError("Pooling masks not yet supported here")
        
    def set_width(self,width):
        if width in ('whole','inf'): width = math.inf
        elif isinstance(width,str): width = int(width)
        self.width = width        
    def set_kernel(self,kern):
        self.kernel = kern
    def set_stride_fraction(self,sf):
        self.stride_fraction = sf
    def set_mesa_fraction(self,mf):
        self.mesa_fraction = mf
    def set_circular(self,circ):
        self.circular = bool(circ)
    def set_pixel_offset(self,offset):
        self.pixel_offset = offset
    def set_boundary_mode(self,mode):
        mode = mode.lower()
        if mode == 'circular': mode = 'wrap'      #convert some older names to their new names
        if mode == 'circular_x': mode = 'wrap_x'
        if mode == 'circular_y': mode = 'wrap_y'
        self.boundary_mode = mode.lower()
    def set_boundary_regions(self,count):
        self.boundary_regions = count
    def set_boundary_forced_pad(self,amount):
        self.boundary_forced_pad = amount
    def set_shrink_factor(self,shrink):
        self.shrink = shrink
        
    def get_width(self):
        return self.width
    def get_stride(self):
        fstride = float(self.width*self.stride_fraction)
        if (fstride < 1) or ((fstride % 1) != 0): raise ValueError(f"Stride_fraction times kernel size must be a positive integer: {self.width} * {self.stride_fraction} = {fstride}")
        stride = int(fstride)
        return stride
        
    def __str__(self):
        retval =  f'{self.width}:kern={self.kernel}:stride={self.stride_fraction}'
        if self.circular != self.DEFAULT_CIRCULAR or self.VERBOSE_PARAMS:
            retval += f':circ={self.circular}'
        if self.mesa_fraction != self.DEFAULT_MESA_FRACTION or self.VERBOSE_PARAMS:
            retval += f':mesa={self.mesa_fraction}'
        if self.pixel_offset is not None and self.pixel_offset != 0:
            retval += f':offset={self.pixel_offset}'
        if self.shrink != self.DEFAULT_SHRINK or self.VERBOSE_PARAMS:
            retval += f':shrink={self.shrink}'
        if self.boundary_regions != self.DEFAULT_BOUNDARY_REGIONS or self.VERBOSE_PARAMS:
            retval += f':extensions={self.boundary_regions}'
        if self.boundary_forced_pad != self.DEFAULT_FORCED_PAD or self.VERBOSE_PARAMS:
            retval += f':pad={self.boundary_forced_pad}'
        return retval
    
    # Create a pooling parameters from a string specifying various pooling parameters
    @classmethod
    def from_str(cls,desc:str):
        fieldname = ''
        index = 0
        retval = PoolingParams(None)
        if desc.startswith('pool='):
            fieldname = 'pool'
            index = 5
        elif desc and desc[0] != ':':
            fieldname = 'pool'
        # convert string into a number (accepts integers, floats, and fractions)
        def _convert_num(s):
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    return Fraction(s)
        # parse and loop over input (could be done more efficiently)
        while index < len(desc):
            c = desc[index]
            index += 1
            if fieldname:
                while (index < len(desc)) and desc[index] != ':': # allow multiple character values
                    c += desc[index]
                    index += 1
                fieldname = fieldname.lower()
                if fieldname == 'pool':
                    retval.set_width(c)
                elif fieldname == 'kern':
                    retval.set_kernel(c)
                elif fieldname == 'stride':
                    val = _convert_num(c)
                    retval.set_stride_fraction(val)
                elif fieldname == 'circ':
                    retval.set_circular(c)
                elif fieldname == 'mesa' or fieldname == 'full': #full is older name for mesa parameter
                    val = _convert_num(c)
                    retval.set_mesa_fraction(val)
                elif fieldname == 'offset':
                    # probably this should be parsed into either a number or tuple pair
                    val = ast.literal_eval(c)
                    retval.set_pixel_offset(val)
                elif fieldname == 'bound':
                    retval.set_boundary_mode(c)
                elif fieldname == 'extensions':
                    val = _convert_num(c)
                    retval.set_boundary_regions(val)
                elif fieldname == 'pad':
                    val = _convert_num(c)
                    retval.set_boundary_forced_pad(val)
                elif fieldname == 'shrink':
                    val = _convert_num(c)
                    retval.set_shrink_factor(val)
                else:
                    raise ValueError(f"Unknown fieldname {fieldname} in {desc}")
                fieldname = ''  # clear the current fieldname
            elif c==':':
                while index<len(desc) and desc[index]!='=':
                    fieldname += desc[index]
                    index += 1
                if desc[index] != '=': raise ValueError(f"PoolingParams field names must end in equals but got {fieldname} in {desc}") 
                if not fieldname: raise ValueError(f"Got empty field name in {desc}")
                index += 1
            elif c==' ' or c=='_':
                pass # skip this character 
            else:
                raise ValueError(f'Unrecognized character "{c}" in {desc}')
        if fieldname: raise ValueError(f'Error: missing value for field {fieldname} in {desc}')
        return retval      
        
    def to_pooling(self):
        if self.mask is not None: raise ValueError(f'mask not yet supported: {self.mask}')
        if isinstance(self.width,(list,tuple)): raise ValueError(f'multi-width (gaze-centric) pooling not yet supported here {self.width}')
        if self.width == math.inf:
            return WholeImagePooling()  #For infinite width kernel use whole region pooling
        if self.mesa_fraction is None:
            self.mesa_fraction = abs(1 - 2*self.stride_fraction)
        kernel = self.kernel.lower()
        shrink = float(self.shrink)
        if kernel == 'trap':
            K = filters.Trapezoid2d(self.width,full_fraction=self.mesa_fraction,circular=self.circular,normalize_area=True,shrink_fraction=shrink)
        elif kernel == 'trig':
            K = filters.Trigezoid2d(self.width,full_fraction=self.mesa_fraction,circular=self.circular,normalize_area=True,shrink_fraction=shrink)
        elif kernel == 'epan':
            K = filters.Epanezoid2d(self.width,full_fraction=self.mesa_fraction,circular=self.circular,normalize_area=True,shrink_fraction=shrink)
        elif kernel == 'box':
            # Full_fraction has no meaning for box filters, use shrink if you want box that only covers part of the kernel region
            K = filters.Box2d(self.width,circular=self.circular,normalize_area=True,shrink_fraction=shrink)
        else:
            raise ValueError(f'Unrecognized kernel type {self.kernel}')
        return RegionPooling(K,self.stride_fraction,pixel_offset=self.pixel_offset,boundary_regions=self.boundary_regions,manual_pad=self.boundary_forced_pad,pad_mode=self.boundary_mode,force_unit_stride=self.force_unit_stride)
            
    # Convert input to standard form, converting any ints or strings to the equivalent PoolingParams objects
    # Input can be single, a list, or a dictionary
    @classmethod 
    def normalize(cls,a):
        if a is None: return a
        if isinstance(a,PoolingParams): return a
        if isinstance(a,str): return cls.from_str(a)  # convert string to PoolingParam
        if isinstance(a,int): return PoolingParams(a)
        if isinstance(a,collections.abc.Sequence):   # convert elements in list/tuple if needed
            for i in range(len(a)):
                if isinstance(a[i],str): a[i] = cls.from_str(a[i])
        if isinstance(a,collections.Mapping):        # convert values in a dictionary if needed
            for key in a.keys():
                if isinstance(a[key],str): a[key] = cls.from_str(a[key])
""#END-CLASS------------------------------------


# Uses the entire image as a single pooling region for statistics 
class WholeImagePooling(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        # This class has no state to initialize 
    
    # Return an image of statistics averaged over each pooling region
    def pool_stats(self,image,original_size=None):
        if image.dim() > 3 and image.size(3)>1:
            return image.mean((-1,-2),keepdim=True)  #if it was a batch image, compute mean only over x and y, but not over batch and channel dimensions
        return image.mean()
    
    def configure_for_downsampling(self,max_dowsampling_factor):
        pass  #nothing to do as we average the whole image regardless of size

    # Return the greatest common divisor of pooling regions strides (useful for knowing how much downsampling can be allowed without creating an invalid stride (stride that is not an integer >= 1)
    def min_stride_divisor(self):
        return 4096  # could be infinite since we have only one window with no stride but 4096 seems like a reasonable conservative finite value to return
    
    # Return a mean image (over roughly pooling region sizes) that can be used for mean subtraction
    def mean_image(self,image):
        return torch.full_like(image,image.mean())  # Return just a constant image of mean value

""#END-CLASS------------------------------------

# Uses a specified kernel to define pooling regions.  Supports overlapping kernels,
# boundary regions (whose support extends partially beyond the image boundary), 
# and can operate on downsampled images (using reduced kernels appropriately)
# One constraint is that kernels must lie on integer coordinates (even in reduced images)
class RegionPooling(torch.nn.Module):
    
    #kernel - tensor representing the pooling region convolutional kernel (ie blur function such as box or gaussian)
    #stride - pixel offset between neighboring kernel centers (if None, then the stride is chosen based on the stride_fraction)
    #stride_fraction - shift in position between neighboring pooling regions (1=no overlap, 1/2=shift half kernel-size between kernels)
    #pixel_offset - offsets all kernel positions by this many pixels (y-offset,x-offset)
    #boundary_regions - number of additional kernels extending (partially) past image boundary (-1 means auto-choose an appropriate number)
    #manual_pad - overrides boundary_regions and uses the specified padding instead
    #pad_mode - padding applied to original image for boundary regions (when kernel extends beyond image boundaries)
    #force_unit_stride - forces the stride to be 1 at all resolutions (only used for special visualization modes where you want the pooled image to match the original image in size)
    def __init__(self,kernel,stride_fraction=1,*,stride=None,pixel_offset=None,boundary_regions=-1,manual_pad=None,pad_mode='zeros',force_unit_stride=False):
        super().__init__()
        if kernel.size(-1) != kernel.size(-2): raise ValueError(f"Non-square kernels not currently supported {kernel.size()}")
        kernelsize = kernel.size(-1)
        if stride is None:
            fstride = float(kernelsize*stride_fraction)
            if (fstride < 1) or ((fstride % 1) != 0): raise ValueError(f"Stride_fraction times kernel size must be a positive integer: {kernel.size()} * {stride_fraction} = {fstride}")
            stride = int(fstride)
        if (stride < 1) or ((stride % 1) != 0): raise ValueError(f"Stride between kernel centers must be a positive integer: {stride}")
        if pixel_offset is None:
            pixel_offset=(0,0)    # Default is no offset
        elif not isinstance(pixel_offset,(list,tuple)): 
            pixel_offset = (pixel_offset,pixel_offset)  #convert to 2-tuple if given as a single number
        if len(pixel_offset)!=2: raise ValueError(f"pixel offset tuple of form (offset_y,offset_x) but got:{pixel_offset}")
        
        # padX = -(pixel_offset[1] % stride)
        # padY = -(pixel_offset[0] % stride)
        # if padX < -stride/2: padX += stride
        # if padY < -stride/2: padY += stride
        # if boundary_regions < 0: boundary_regions = math.ceil(kernelsize/(2*stride))       # Default is extend boundary by half the kernel size
        # if stride >= kernelsize: boundary_regions = 0     # Cannot have boundary regions if regions have no overlap
        # if boundary_regions > math.ceil(kernelsize/stride - 1): raise ValueError(f"too many boundary regions, must at least overlap image partially {boundary_regions} > {math.ceil(1/stride_fraction - 1)}")
        # padX += stride*boundary_regions
        # padY += stride*boundary_regions
        # if padX < 0: padX += stride
        # if padY < 0: padY += stride
        
        # padMinX = -(pixel_offset[1] % stride)
        # padMinY = -(pixel_offset[0] % stride)
        # padMaxX = 0
        # padMaxY = 0
        # print(f'padMinX {padMinX}  padMinY {padMinY}')
        # if boundary_regions < 0: boundary_regions = math.ceil(kernelsize/(2*stride))       # Default is extend boundary by half the kernel size
        # if stride >= kernelsize: boundary_regions = 0     # Cannot have boundary regions if regions have no overlap        
        # if pad_mode != 'wrap_x' and pad_mode != 'wrap':
        #     padMinX += stride*boundary_regions
        #     padMaxX += stride*boundary_regions
        # if pad_mode != 'wrap_y' and pad_mode != 'wrap':
        #     padMinY += stride*boundary_regions
        #     padMaxY += stride*boundary_regions
        # if padMinX < 0: padMinX += stride
        # if padMinY < 0: padMinY += stride
        
        if boundary_regions < 0: boundary_regions = math.ceil(kernelsize/(2*stride))       # Default is extend boundary by half the kernel size
        if stride >= kernelsize: boundary_regions = 0     # Cannot have boundary regions if regions have no overlap        
        self.boundary_regions = boundary_regions
        self.manual_pad = manual_pad
        self.pixel_offset = pixel_offset
        self.base_image_width = None
        self.base_image_height = None

        if (kernel.dim()==2):     # Pytorch prefers 4d tensors (batch,channel,height,width)
            kernel = kernel.unsqueeze(0).unsqueeze(0)
        elif (kernel.dim()==3):
            kernel = kernel.unsqueeze(0)
        def _wrap(filt):
            return torch.nn.Parameter(filt,requires_grad=False)
        self.base_stride = stride
        # self.base_padMinX = padMinX
        # self.base_padMinY = padMinY
        # self.base_padMaxX = padMinX
        # self.base_padMaxY = padMaxY
        self.pad_mode = pad_mode
        self.force_unit_stride = force_unit_stride  # Used for visualization purposes only, forces stride to be one regardless of level etc.
        # The kernel is kept as first item in a list which may later also contain downsampled versions
        self.kernels = torch.nn.ParameterList()     # Use parameter list so all kernels are recognized as parameters (for GPU etc)
        self.kernels.append(_wrap(kernel))
        
    # Configure this pooling object to be able to process downsampled images (up to some maximum factor)
    def configure_for_downsampling(self,max_downsampling_factor):
        if len(self.kernels) != 1:
            raise ValueError('Downsampling was already configured for this pooling object')
        def _wrap(filt):
            return torch.nn.Parameter(filt,requires_grad=False)
        kernel = self.kernels[0]
        factor = 1
        maxlevel = 0
        while factor < max_downsampling_factor:
            factor = 2*factor
            maxlevel += 1
            if (kernel.size(-1)%factor!=0) or (kernel.size(-2)%factor!=0):
                raise ValueError(f'Kernel size must be a multiple of downsamping factor {factor}')
            down = 4*F.avg_pool2d(kernel,factor)  # Downsample kernel by factor of 2 via averaging
            self.kernels.append(_wrap(down))
        if factor != max_downsampling_factor:
            raise ValueError(f'Max downsampling factor must be a power of two {max_downsampling_factor}')
        if (self.base_stride>>maxlevel)<<maxlevel != self.base_stride:
            raise ValueError(f"Max downsampling factor {factor} must be a divisor of pooling stride {self.base_stride}")
        if (self.pixel_offset[0]>>maxlevel)<<maxlevel != self.pixel_offset[0] or (self.pixel_offset[1]>>maxlevel)<<maxlevel != self.pixel_offset[1]:
            raise ValueError(f"Max downsampling factor {factor} must be a divisor of pixel offset {self.pixel_offset}")
            
    def _configure_padding(self,width,height):
        stride = self.base_stride
        pad_mode = self.pad_mode
        boundary_regions = self.boundary_regions
        kernX = self.kernels[0].size(-1)
        kernY = self.kernels[0].size(-2)
        padMinX = -(self.pixel_offset[1] % stride)
        padMinY = -(self.pixel_offset[0] % stride)
        if pad_mode != 'wrap_x' and pad_mode != 'wrap':
            padMinX += stride*boundary_regions
            padMaxX = stride*boundary_regions
            if padMinX < 0: padMinX += stride
            if (width+padMinX+padMaxX-kernX) % stride > min(stride//2,padMaxX): padMaxX += stride  #Add extra region if some pixels will not be covered or if more than half a stride will be leftover/unused
        else:
            if padMinX < 0: padMinX += stride
            padMaxX = kernX - 1 - ((width-1)%stride) - padMinX
            if padMaxX < 0: padMaxX = 0  # F.pad does not like negative padding values
            
        if pad_mode != 'wrap_y' and pad_mode != 'wrap':
            padMinY += stride*boundary_regions
            padMaxY = stride*boundary_regions
            if padMinY < 0: padMinY += stride
            if (height+padMinY+padMaxY-kernY) % stride > min(stride//2,padMaxY): padMaxY += stride  #Add extra region if some pixels will not be covered or if more than half a stride will be leftover/unused
        else:
            if padMinY < 0: padMinY += stride
            padMaxY = kernY - 1 - ((height-1)%stride) - padMinY
            if padMaxY < 0: padMaxY = 0  # F.pad does not like negative padding values
        if self.manual_pad is not None:  # if user is overriding the usual automatic padding, then just use their manually specified value
            padX = padY = self.manual_pad
            if padX>0 and padX<1: # if it is a fraction then assume the amount is specified relative to the kernel size
                padX = float(padX*kernX)
                padY = float(padY*kernY)
                if (padX!=int(padX)) or (padY!=int(padY)): raise ValueError('Pooling padding must be an integer but got {padX},{padY} from {self.manual_pad}')
                padX = int(padX)
                padY = int(padY)
            padMinX = padMaxX = padX
            padMinY = padMaxY = padY
#        print(f' padX {padMinX},{padMaxX}  padY {padMinY},{padMaxY}')
        self.padMinX = padMinX
        self.padMaxX = padMaxX
        self.padMinY = padMinY
        self.padMaxY = padMaxY
        self.base_image_width = width
        self.base_image_height = height
        
    def pool_stats(self,image,original_size):
        # If image was downsampled, we need to select the appropriately downsampled kernel to use with it
        original_width = original_size[-1]
        original_height = original_size[-2]
        if original_width != self.base_image_width or original_height != self.base_image_height:
            # If first time seeing image or image size has changed, we need to compute the appropriate padding values (which may depend on the image size)
            self._configure_padding(original_width, original_height)
        width = image.size(-1)
        height = image.size(-2)
        level = original_width.bit_length() - width.bit_length()
        if width<<level != original_width or height<<level != original_height:
            raise ValueError(f"image downsampling only supported for powers of two {image.size(-1)} vs {original_width}")
        region_kernel = self.kernels[level]
        # Adjust stride and padding for downsampling 
        stride = self.base_stride>>level   
        padMinX = self.padMinX>>level
        padMinY = self.padMinY>>level
        padMaxX = self.padMaxX>>level
        padMaxY = self.padMaxY>>level
        if self.force_unit_stride: stride = 1
        
        if self.pad_mode == 'zeros':
            if padMinX!=padMaxX or padMinY!=padMaxY:  # need to use separate call to pad if the padding is not symmetric
                image = F.pad(image,(padMinX,padMaxX,padMinY,padMaxY))
                padMinX = padMinY = 0
            stat = F.conv2d(image,region_kernel,padding=(padMinY,padMinX),stride=stride)
        elif self.pad_mode == 'wrap' or self.pad_mode == 'circular':
            stat = F.conv2d(F.pad(image, (padMinX,padMaxX,padMinY,padMaxY),mode='circular'), region_kernel,stride=stride)
        elif self.pad_mode == 'wrap_x' or self.pad_mode == 'circular_x':
            if padMinY!=padMaxY: # need to use separate call to pad if the padding is not symmetric
                image = F.pad(image,(0,0,padMinY,padMaxY))
                padMinY = 0
            stat = F.conv2d(F.pad(image, (padMinX,padMaxX,0,0),mode='circular'), region_kernel,padding=(padMinY,0),stride=stride)
        elif self.pad_mode == 'wrap_y' or self.pad_mode == 'circular_y':
            if padMinX!=padMaxX: # need to use separate call to pad if the padding is not symmetric
                image = F.pad(image,(padMinX,padMaxX,0,0))
                padMinX = 0
            stat = F.conv2d(F.pad(image, (0,0,padMinY,padMaxY),mode='circular'), region_kernel,padding=(0,padMinX),stride=stride)
        else:
            raise ValueError(f'Unsupported pad_mode {self.pad_mode}')
        return stat
        
    # This method takes a pooled state image and interpolates/splats them into a higher resolution image
    # Note: this is not the inverse of pool_stats() but can be very useful for approximating higher resolution stat images
    def blame_stats(self,stat_image,original_size):
        raise ValueError('This routine is out of date and needs to be updated before it can be used again')
        region_kernel = self.kernels[0]        
        stride = self.base_stride          # Spacing between regions
        padX = self.base_padX
        padY = self.base_padY
        original_width = original_size[-1]
        original_height = original_size[-2]
        if ((original_width+2*padX)%stride) > padX: padX += stride   # Make sure the entire image will be covered by pooling regions
        if ((original_height+2*padY)%stride) > padY: padY += stride  # Make sure the entire image will be covered by pooling regions
        img = F.conv_transpose2d(stat_image,region_kernel,padding=(padY,padX),stride=stride)
        return (stride*stride)*img    
    
    def kernel_size(self):
        return self.kernels[0].size(-1)

    # Return the greatest common divisor of pooling regions strides (useful for knowing how much downsampling can be allowed without creating an invalid stride (stride that is not an integer >= 1)
    def min_stride_divisor(self):
        # largest factor that even divides into the stride and the padding
        return math.gcd(math.gcd(self.base_stride,self.kernel_size()),math.gcd(self.pixel_offset[0],self.pixel_offset[1]))

""#END-CLASS------------------------------------            
 
# Code to generate various specific regions kernels.  Could be implemented as subclasses in the future

# Region pooling using a simple box function
def Box(pooling_size,stride_fraction=1/4,*,circular=False,pixel_offset=None,boundary_regions=-1,pad_mode='zeros'):
    K = filters.Box2d(pooling_size,circular=circular,normalize_area=True)
#    K = torch.ones(pooling_size,pooling_size)
#    K = K / (K.sum())  # Normalize kernel
    return RegionPooling(K,stride_fraction,pixel_offset=pixel_offset,boundary_regions=boundary_regions,pad_mode=pad_mode)

# Trapezoid shaped kernel (eg, constant for a middle segments with linear ramps at first and last segments)
# Number of segments depends on the overlap setting (so that overlapped kernels will sum to a constant value)
def Trapezoid(pooling_size,stride_fraction=1/4,*,circular=False,pixel_offset=None,boundary_regions=-1,pad_mode='zeros'):
    if stride_fraction >= 1: raise ValueError('stride_fraction must be at <1 (use Box for constant region with no overlap or unit stride)')
    full_fraction = abs(1 - 2*stride_fraction)
    K = filters.Trapezoid2d(pooling_size,full_fraction=full_fraction,circular=circular,normalize_area=True)
    if False:
        plot_image(K)
        end = 1 - (1.0/pooling_size)
        # Create coordinates in range [-1,1] for support region of kernel
        x = torch.linspace(-end,end,steps=pooling_size)
        # Kernel has overlap segments, construct linear slopes for first and last segments
        if stride_fraction <= 1/2:
            endslope = float(1/(2*stride_fraction))
        else:
            endslope = float(1.0/(2 - (2*stride_fraction)))
        kern1d = torch.min(endslope*(1+x),endslope*(1-x))
        #kern1d = torch.min((2+2*x),(2-2*x)) # Create trapezoid
        # Clamp middle segments to have constant value 1 (before normalization)
        kern1d = kern1d.clamp(max=1)
        # Construct 2d kernel as outer product of the 1d kernels
        K = torch.ger(kern1d,kern1d)   # 2d Kernel is outer product of 1d kernels
        K = K / (K.sum())  # Normalize kernel
        plot_image(K)
    return RegionPooling(K,stride_fraction,pixel_offset=pixel_offset,boundary_regions=boundary_regions,pad_mode=pad_mode)

# Similar to trapezoid filter except uses trigonometric blending function that is C2 continuous
def Trigezoid(pooling_size,stride_fraction=1/4,*,circular=False,pixel_offset=None,boundary_regions=-1,**kwargs):
    if stride_fraction >= 1: raise ValueError('stride_fraction must be at <1 (use Box for constant region with no overlap or unit stride)')
    full_fraction = abs(1 - 2*stride_fraction)
    K = filters.Trigezoid2d(pooling_size,full_fraction=full_fraction,circular=circular,normalize_area=True)
    if False:
        plot_image(K)
        end = 1 - (1.0/pooling_size)
        # Create coordinates in range [-1,1] for support region of kernel
        x = torch.linspace(-end,end,steps=pooling_size)
        # Kernel has overlap segments, construct linear slopes for first and last segments for angles
        # Angle will be zero in the middle and slope up to pi/2 linearly in the end segments
        if stride_fraction <= 1/2:
            endslope = float(1/(2*stride_fraction))
        else:
            endslope = float(1.0/(2 - 2*stride_fraction))
        angle1d = torch.max((endslope*(x-1))+1,(endslope*(-x-1))+1).clamp_(min=0)
        # Taking cosine^2 of angle produces kernel that is one in the middle segments and blends smoothly to zero at edges
        kern1d = torch.cos((math.pi/2)*angle1d)**2
        # Construct 2d kernel as outer product of the 1d kernels
        K = torch.ger(kern1d,kern1d)   # 2d Kernel is outer product of 1d kernels
        K = K / (K.sum())  # Normalize kernel
        plot_image(K)
    return RegionPooling(K,stride_fraction,pixel_offset=pixel_offset,boundary_regions=boundary_regions,**kwargs)    
        
# Pools image statistics using multiple pooling objects/functions. 
#Weights and accumulates resulting statistics into a combined result
# Provides a way to simulate having the pooling function vary over the image
class RegionPoolingList(torch.nn.Module):
    
    def __init__(self,regionpooling_list,regionweights=None):
        super().__init__()
        # Stored in a module list so that pytorch can find all the sub-modules
        self.pool_list = torch.nn.ModuleList(regionpooling_list)
        if regionweights is None:
            self.weights = None
        else:
            self.weights = torch.nn.ParameterList()
            for w in regionweights:
                self.weights.append(torch.nn.Parameter(w,requires_grad=False))
        
    def configure_for_downsampling(self,max_downsampling_factor):
        for p in self.pool_list:
            p.configure_for_downsampling(max_downsampling_factor)
        
    # Return an image of statistics averaged over each pooling region
    def pool_stats(self,image,original_size):
        stat_list = []
        for i,p in enumerate(self.pool_list):
            stat = p.pool_stats(image,original_size)
            #print(f"image {stat.size()}  weight {self.weights[i].size()}")
            if self.weights is not None: stat = stat*self.weights[i]
            stat_list.append(stat.view(-1))
        return torch.cat(stat_list)
        
    # Return the greatest common divisor of pooling regions strides (useful for knowing how much downsampling can be allowed without creating an invalid stride (stride that is not an integer >= 1)
    def min_stride_divisor(self):
        mindivisor = 0    # note: gcd(0,a)==a for any positive integer a
        for p in self.pool_list:
            minval = math.gcd(mindivisor,p.min_stride_divisor())
        return minval

""#END-CLASS------------------------------------
    
# Given an image where each pixel has the desired pool_size and a list of region pooling kernels
# Build combined pooling list where each kernel is restricted to image regions where its
# kernel size is equal to or larger than the local desired pool_size
# poolsize_image must be the same size as the images it will be used on
def VariableRegionPooling(poolsize_image,kernel_list):
    while poolsize_image.dim() < 4: poolsize_image = poolsize_image.unsqueeze(0)
    ones = torch.ones_like(poolsize_image)
    k_list = []
    w_list = []
    # Compute the weighting matrix for each kernel
    for k in kernel_list:
        kern_size = k.kernel_size()
        # Compute a matrix containing the total weight for each pooling region with this kernel
        norm = k.pool_stats(ones,ones.size(-1))
        # Then build matrix with average pooling size for each region
        avg_poolsize = k.pool_stats(poolsize_image,poolsize_image.size(-1)) / norm
        weight = torch.le(avg_poolsize,kern_size)*torch.ge(avg_poolsize,kern_size/2)
        weight = weight.float() / norm  # Renormalize any kernels that did not sum to one (eg due to boundary effects)
        weight = weight / norm.sum().sqrt() # Partially renormalize for number of kernels in this level
        #print(weight)
        # Only add kernel to list if it is used at least somewhere in the image
        if (weight.sum() > 0):
            k_list.append(k)
            w_list.append(weight)
    return RegionPoolingList(k_list,w_list)


# Some utility data structures for specifying the gaze point in various ways
# A record containing a point in pixel coordinates (note unlike torch.tensor coordinates, x is before y here)
class ImagePoint(NamedTuple):
    x : float        
    y : float  
    def get_in_torch_coordinates(self,target_image=None):  # target_image is unused here but is needed for NormalizedPoint below
        return (self.y,self.x)    # torch represents images as channelXheightXwidth so y value comes before the x component
""#END-CLASS------------------------------------    
      
# A record containing a point in normalized coordinates, between 0 and 1 (as opposed to pixel coordinates)
class NormalizedPoint(NamedTuple):
    x : float        
    y : float        
    def get_in_torch_coordinates(self,target_image):
        if torch.is_tensor(target_image): size = target_image.size()
        else: size = target_image  # target image can be an image or just an imagesize object
        return (self.y*(size[-2]-1), self.x*(size[-1]-1))  #pixels are on integer coordinates starting at zero
""#END-CLASS------------------------------------    

# Create a pooling object based on specified size(s) and some default parameters
# Pooling size is a single number specifying the size/diameter for the pooling regions
def make_uniform_pooling(pooling_size,stride_fraction=1/4,pixel_offset=None):
    # handle case where input is a size list instead of a number (list with just one number is okay)
    if isinstance(pooling_size,(tuple,list)):
        if len(pooling_size) != 1:
            raise RuntimeError(f'Uniform pooling should have use a single size: {pooling_size}')
        pooling_size = pooling_size[0]
    if pooling_size >=1e15:            # extremely large pooling indicates use whole image pooling
        return WholeImagePooling()
    if stride_fraction >= 1: 
        return Box(pooling_size,stride_fraction,pixel_offset=pixel_offset) # USe box kernel if there is no overlap between the kernels
    return Trigezoid(pooling_size,stride_fraction,pixel_offset=pixel_offset)
    
# Create a combination of pooling regions based on a specified set of pooling region sizes, a gaze point, and an eccentricity scaling factor
# A gaze-centric pooling, uses different size pooling regions depending on the distance from the gaze point
# Note: gaze point is in torch.tensor coordinates (y before x and in units of pixels)
# Returns tuple of (poolingobject,copymask) where the pooling object implements the pooling regions and the copymask
# specifies the (gaze-centered) region that should just be copied verbatim from the reference (or None if region is empty)
def make_gaze_centric_pooling(pooling_sizes,target_image,gazepoint,
                              eccentricity_scaling=0.5,stride_fraction=1/4, pixel_offset=None):
    if isinstance(pooling_sizes,(int,float)): pooling_sizes = [pooling_sizes]    #convert to list if pooling_size was given as a single number
    # if no gaze point was given or there is only one pooling size return uniform (same) pooling everywhere 
    if (gazepoint is None) or (len(pooling_sizes)==1):
        if len(pooling_sizes) != 1: raise RuntimeError(f'Uniform pooling should have just a single size: {pooling_sizes}')
        if pooling_sizes[0] == 0:  #special flag value meaning just copy the reference image exactly
            return (WholeImagePooling(),torch.ones_like(target_image,dtype=torch.bool))
        # otherwise use uniform size pooling and no copy region
        return (make_uniform_pooling(pooling_sizes[0],stride_fraction,pixel_offset), None)


    if pixel_offset is not None:
        raise ValueError(f"Pooling offsets not yet supported for gaze-centric pooling")
    # for gaze-centric, we need to construct a set of pooling ovjects for each specified size
    # and masks for which parts of the image they are to be used    
    # Make an image of the target per-pixel pooling size based on eccentricity from gaze point
    pooling_size_image = eccentricity_scaling*(blend.distance_image(target_image,gazepoint).unsqueeze(0).unsqueeze(0))
    # make an image of all ones, that we use to compute normalization factors
    ones_image = torch.ones_like(pooling_size_image)
    pooling_sizes.sort(reverse=True)  #sort in descending order
    prev_size = math.inf  
    kernlist = []
    weightlist = []
    copymask = None
    # Convert gaze point to coordinates used in torch
    gazepoint = gazepoint.get_in_torch_coordinates(target_image)
    for size in pooling_sizes:   # iterate through pooling sizes from biggest to smallest
        if size == 0:            # pooling size of zero is used mean we want a copy-exact region near the gaze point
            copymask = torch.lt(pooling_size_image,prev_size)
            print(f"copy exact for 0 <= value < {prev_size}")
            break
        # create a pooling object with the corresponding size pooling regions
        kern = Trigezoid(size,stride_fraction)
        # use this pooling for regions whose mean target size is < prev_size and >= 0.75*size
        # could use >= target size, but we extend it a little extra to ensure there is some overlap between regions for neighboring sizes
        start = size - (size/4)
        stop = prev_size
        if size == pooling_sizes[0]: start = 0  #if this is smallest size, then don't limit minimum target size of when to apply it
        print(f"pool {size} for {start} <= value < {stop}")
        
        # Compute a matrix containing the total weight for each pooling region within this kernel 
        norm = kern.pool_stats(ones_image,ones_image.size())
        # and build matrix with average target pooling size for each region
        avg_poolsize = kern.pool_stats(pooling_size_image,pooling_size_image.size()) / norm
        # construct mask matrix of which regions of this size to use (ie those that fit the mean target size constraints above)
        weight = torch.ge(avg_poolsize,start)*torch.lt(avg_poolsize,stop)
#        weight = weight.float() / norm  # Renormalize any kernels that did not sum to one (eg due to boundary effects)
        weight = weight / norm.sum().sqrt() # Partially renormalize for number of kernels in this level
        #print(weight)
        prev_size = size;
        # Only add kernel to list if it is used at least somewhere in the image
        if (weight.sum() > 0):
            kernlist.append(kern)
            weightlist.append(weight)
    
    # construct the combined size pooing object and return it along with the copy mask
    pooling = RegionPoolingList(kernlist,weightlist)
    return (pooling,copymask)        
    
def _testmain():
    k = Trigezoid(64,stride_fraction=3/4,pixel_offset=(0,0))
    k2 = Trapezoid(64,stride_fraction=1/2,pixel_offset=(16,0))
    k3 = Box(64,stride_fraction=1,pixel_offset=(0,0))
#    plot_image(k.kernels[0])
#    plot_image(k2.kernels[0])
    image = torch.rand(1,1,128,256)
    pooled = k2.pool_stats(image,image.size())
#    plot_image(image)
#    plot_image(pooled)
    
def _testcircular():
    wh = torch.ones(1,1,32,32)
    bl = torch.zeros(1,1,32,32)
    image = torch.cat([torch.cat([wh,wh], dim=-1),torch.cat([wh,bl],dim=-1)],dim=-2)
    plot_image(image,title='image')
    for pm in ['zeros','circular','circular_x','circular_y']:
        k = Trigezoid(4,pad_mode=pm)
        plot_image(k.pool_stats(image,image.size()),title=pm)
    
    
  
if __name__ == "__main__":   # execute main() only if run as a script
    _testmain()
#    _testcircular()
    
    