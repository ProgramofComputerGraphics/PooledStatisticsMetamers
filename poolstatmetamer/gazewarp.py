# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:45:34 2021

Code to perform forward and inverse log-polar transform of an image.  For gaze-centric images
the pooling regions should grow in size linearly with distance from the gaze-point, but our
code only natively supports uniformly-sized pooling regions.  The idea of the log-polar
transform is to warp the image such that the size and shapes of all the pooling regions are
equalized.  Then we can compute a uniform metamer on the warped image and then unwarp it to
get a gaze-centric metamer.

The log-polar tranform is highly non-homogeneous and some parts of the image will be streched
while others are shrunk.  This causes some loss of information and possible aliasing.  By 
default we apply some pre-blurring to the image before warping it to reduce the artifacts
from aliasing/under-sampling.  Note we only preblur in the forward warp but not the inverse/unwarp.
We assume warped images will either have been generated through the warp process or have
matched statistics to one that did and thus not contain much frequency content beyond
what the inverse/unwarp can handle (so aliasing is less of a concern in this case).

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import math
import torch
import torch.nn.functional as F
import os
import time
from fractions import Fraction
from typing import Optional
import imageblends as blend
import poolingregions as pool

# A record containing parameters for log-polar warp used to equalize gaze-centric pooling regions
class WarpParams():
    #TODO: test gamma setting across a wider range of displays/systems
    #TODO: define better ways to set the minimum radius
    
    # Some default values, hopefully these will only rarely be changed
    DEFAULT_SCALING = 0.5
    DEFAULT_TARGET_POOLING_SIZE = None  # None means automatically set to match pooling region size
    DEFAULT_ANISOTROPY_RATIO = 1
    DEFAULT_MIN_RADIUS = None        # None means automatically choose a min radius based on the other parameters
    DEFAULT_AZIMUTH_MULTIPLE = True  # Set azimuthal size to be multiple of given value (or to autoset value if True)
#    DEFAULT_ROUND_AZIMUTH = True
    DEFAULT_GAMMA = 1.6          # Gamma values range from ~1.6 to 2.6 and can vary with monitor, viewing angle, ambient lighting, viewing software, etc.
                                 # Currently we are defaulting to low value to avoid over-gamma correcting
    DEFAULT_CLAMP_RANGE = True   # Clamp output images to the range [0,1] in case interpolation might exceed this range
    VERBOSE_PARAMS = False  # print parameter values even when they match the default values?    
    
    def __init__(self,scaling=DEFAULT_SCALING,target_pooling_size=DEFAULT_TARGET_POOLING_SIZE,*, anisotropy=DEFAULT_ANISOTROPY_RATIO, min_radius=DEFAULT_MIN_RADIUS, azimuth_multiple=DEFAULT_AZIMUTH_MULTIPLE, gamma=DEFAULT_GAMMA, clamp_range=DEFAULT_CLAMP_RANGE):
        super().__init__()
        self.scaling             :float # scaling of pooling region size with eccentricity (ie the ratio pooling_size/eccentricity)
        self.target_pooling_size :int # size of pooling regions in the warped image (pooling regions have constant size in warped image)
        self.anisotropy          :float # Anisotropy ratio for pooling region shapes (1 = circular)
        self._min_radius         :Optional[int]  # Minimum radius to appear in warped image.  Must be >0 or warp image would be infinitely large
        self.azimuth_multiple    :Optional[int|bool]
#        self.round_azimuth       :bool  # Force azimuth resolution to be a multiple of target_pooling_size
        self.gamma               :Optional[float] # Gamma value input image is encoded with (will transform to linear space for warping operations and then back to gamma)
        self.clamp_range         :bool  # clamp generated images to range [0,1] 
        # initialize parameters
        self.scaling = scaling
        self.target_pooling_size = target_pooling_size
        self.anisotropy = anisotropy
        self.azimuth_multiple = azimuth_multiple  # Force angular resolution to be a multiple of the given value (True==pooling_stride)
        self._min_radius = min_radius
        self.gamma = gamma
        self.clamp_range = clamp_range
        
    def set_warp_scaling(self,val):
        if val <=0 or val >1.8: raise ValueError(f'Invalid value for warp scaling {val}  Note:must be in range (0,1.8] and warped pooling region sizes grow increasingly inaccurate as scale approaches 2')
        self.scaling = val
        
    def set_anisotropy_ratio(self,val): self.anisotropy = val
    def set_target_pooling_size(self,val): self.target_pooling_size = val
    def set_min_radius(self,val): self._min_radius = val
        
    # Used by metamer solver to suggest pooling size (if not already set) and stride or print error/warning if mismatched with what was already set in this warp
    def suggest_pooling_params(self,pooling_params):
        if isinstance(pooling_params,pool.PoolingParams):
            size = pooling_params.get_width()
            stride = pooling_params.get_stride()
        else:
            raise ValueError(f'Expected pooling parameters object but got {type(pooling_params)}')
        if self.target_pooling_size is None:
            if not isinstance(size, (int,float)): raise ValueError(f'Expected a number but got {size}')
            self.target_pooling_size = size
        elif self.target_pooling_size != size:
            # Not sure if this should be an error or just a warning, but making it an error for now until there is a reason to allow it
            raise ValueError(f'Mismatch between warp and actual pooling size {self.target_pooling_size} vs {size}.  Would change eccentricity scaling')
            print('WarpParams: Target pooling already set to {self.target_pooling_size} so ignoring suggestion of {size}')
        if self.azimuth_multiple is True:  #Auto set azimuth multiple to be the pooling stride if requested
            if not isinstance(stride, int): raise ValueError(f'Expected integer stride but got {stride}')
            self.azimuth_multiple = stride
        elif self.azimuth_multiple:
            if self.azimuth_multiple/stride % 1 != 0: print(f'Warning: azimuth multiple {self.azimuth_multiple} is not a multiple of the pooling stride {stride} in gaze warp')
            
    def get_min_radius(self):
        if self._min_radius is None and self.target_pooling_size is not None:
            # if no minimum radius has been specified then default to a minimum of 1/4 of the target pooling size
#            return self.target_pooling_size/6
             return self.target_pooling_size/4
#            return self.target_pooling_size/self.scaling - self.target_pooling_size/2
        return self._min_radius    
    
    def __str__(self):
        retval = f'warp={self.self.ecc_scaling}'
        if self.anisotropy != self.DEFAULT_ANISOTROPY_RATIO or self.VERBOSE_PARAMS:
            retval += f':anisotropy={self.anisotropy}' 
        if self.target_pooling_size != self.DEFAULT_TARGET_POOLING_SIZE or self.VERBOSE_PARAMS:
            retval += f':target_size={self.target_pooling_size}'
        if self._min_radius != self.DEFAULT_MIN_RADIUS or self.VERBOSE_PARAMS:
            retval += f':min_radius={self._min_radius}'
        return retval
    
    # Create a warp parameters from a string specifying various warping parameters
    @classmethod
    def from_str(cls,desc:str):
        fieldname = ''
        index = 0
        retval = WarpParams(None)
        if desc.startswith('warp='):
            fieldname = 'warp'
            index = 5
        elif desc and desc[0] != ':':
            fieldname = 'warp'
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
                if fieldname == 'warp':
                    retval.set_warp_scaling(_convert_num(c))
                elif fieldname == 'anisotropy':
                    retval.set_anisotropy_ratio(_convert_num(c))
                elif fieldname == 'target_pooling_size':
                    retval.set_target_pooling_size(_convert_num(c))
                elif fieldname == 'min_radius':
                    retval.set_min_radius(_convert_num(c))
                else:
                    raise ValueError(f"Unknown fieldname {fieldname} in {desc}")
                fieldname = ''  # clear the current fieldname
            elif c==':':
                while index<len(desc) and desc[index]!='=':
                    fieldname += desc[index]
                    index += 1
                if desc[index] != '=': raise ValueError(f"WarpParams field names must end in equals but got {fieldname} in {desc}") 
                if not fieldname: raise ValueError(f"Got empty field name in {desc}")
                index += 1
            elif c==' ' or c=='_':
                pass # skip this character 
            else:
                raise ValueError(f'Unrecognized character "{c}" in {desc}')
        if fieldname: raise ValueError(f'Error: missing value for field {fieldname} in {desc}')
        return retval      

    # Convert input to standard form, converting any ints or strings to the equivalent WarpParams objects
    # Input can be single, a list, or a dictionary
    @classmethod 
    def normalize(cls,a):
        if a is None: return a
        if isinstance(a,WarpParams): return a
        if isinstance(a,str): return cls.from_str(a)  # convert string to PoolingParam
        if isinstance(a,float): return WarpParams(a)

""#END-CLASS------------------------------------

#--Some utlity functions used by the warping code----------------------------
# return ceil but also required to be a multiple of the given factor
def _ceil_multiple(x,factor):
    return factor*math.ceil(x/factor)

def _max_radius(image,gaze):
    if torch.is_tensor(image): 
        x = max(gaze[-1],image.size(-1)-gaze[-1])
        y = max(gaze[-2],image.size(-2)-gaze[-2])
    else:
        x = max(gaze[-1],image[-1]-gaze[-1])
        y = max(gaze[-2],image[-2]-gaze[-2])    
    x += 0.5
    y += 0.5
    return math.sqrt(x*x + y*y)

# Computes distance image (distance to center point for each pixel) plus the x_only_distance and y_only_distance images
def _make_distance_image(image_size,center,device=None):
    if torch.is_tensor(image_size):
        width = image_size.size(-1)
        height = image_size.size(-2)
    elif isinstance(image_size,(torch.Size,tuple,list)):
        width = image_size[-1]
        height = image_size[-2]
    # adjusting to coordinates where integers are pixel corners and middle is +.5
    x = torch.arange(0.5,width,device=device) - (center[-1]+0.5)
    y = torch.arange(0.5,height,device=device) - (center[-2]+0.5)
    x2 = (x**2).unsqueeze(0)
    y2 = (y**2).unsqueeze(1)
    dist2 = x2 + y2
    return torch.sqrt(dist2),x.unsqueeze(0).expand(dist2.size()),y.unsqueeze(1).expand(dist2.size())

def _preblur_image(image,level,gamma=None):
    if image.dim() != 4: raise ValueError(f'expected 4d tensor {image.size()}')
    if level != 2: raise ValueError(f'not yet implemented {level}')
    filt1d = torch.tensor([0.25,0.5,0.25]) # a very basic box/trapezoid filter for now, should be improved and generalized
    # create separable filter as outer product plus expand to pytorch's preferred 4d tensor format: batch x channel x height x width
    filt = torch.outer(filt1d,filt1d)[None,None,:,:]
    filt = filt.expand(image.size(1),-1,-1,-1)  # expand filter to have the same number of channels as the input image
    if filt.size(-1) != filt.size(-2) or (filt.size(-1) % 2 != 1): raise ValueError(f'filter must be squared and size must be odd {filt.size()}')
    pad = (filt.size(-1)-1)//2   #we need to add padding so convolution does not change its size
    if gamma:
        image = image.pow(gamma)
    # We use reflect-mode padding to avoid darkening at the boundary, but means we have to manually pad the input
    res = F.conv2d(F.pad(image,(pad,pad,pad,pad),mode='reflect'), filt,groups=image.size(1))
    if gamma:
        res = res.pow(1/gamma)
    return res

def _make_mipmap_filter2d(level,filter_type):
    if filter_type == 'box':
        # simple box filter centered on the pixel with width equal to 2^level
        if level <= 0: return torch.ones(1,1)
        filt1d = torch.ones(2**level+1)
        filt1d[0] = filt1d[-1] = 0.5  # box filter extends halfway into the filter's end pixels
        filt1d = filt1d/filt1d.sum() # normalize filter
        return torch.outer(filt1d,filt1d) # create 2d separable filter from 1d version
    else:
        raise ValueError('Unrecognized mipmap filter type {filter_type}')
        
# Build a mipmap-style pyramid of increasingly blurred images
# Note: all levels use the same image resolution, unlike a traditional mipmap where images would decrease (shrink) with increasing level
# Filtering is imperfect and there can still be undersampling errors, especially at higher levels where filter shape could be further optimized
def _build_mipmap(image,level_count,mode='box'):
    if image.dim() != 4: raise ValueError(f'expected 4d tensor {image.size()}')
    mipmap = [None]*level_count
    mipmap[0] = image #image forms first level of mip map
    ones = torch.ones(1,1,image.size(2),image.size(3), device=image.device)
    for level in range(1,level_count):
        filt = _make_mipmap_filter2d(level,mode)
        if image.is_cuda: filt = filt.cuda()
#        plot_image(filt,title=f'mipmap filter for level {level}')
        filt = filt[None,None,:,:] # convert 2d tensor for standard 4d format
        filt = filt.expand(image.size(1),-1,-1,-1)  # expand (repeat) filter to have the same number of channels as the input image
        if filt.size(-1) != filt.size(-2) or (filt.size(-1) % 2 != 1): raise ValueError(f'filter must be squared and size must be odd {filt.size()}')
        pad = (filt.size(-1)-1)//2   #we need to add padding so the convolution does not change its size
        blurred = F.conv2d(image, filt, padding=pad, groups=image.size(1))
        norm = F.conv2d(ones, filt.narrow(1,0,1), padding=pad)  #normalization factor to correct boundary darkening
        mipmap[level] = blurred / norm
#        plot_image(mipmap[level],f'mipmap level {level}')
    return mipmap
       
#TODO: Test better mipmap construction filters (jinc?) and dynamic number of levels
def _preblur_image_mipmap(image,level_map):
#    num_levels = 5
    num_levels = math.ceil(level_map.max())
#    print(f"num mipmap levels {num_levels}")
    mipmap = _build_mipmap(image,num_levels)  
    retval = mipmap[0]
    for lvl in range(1,num_levels):
        blend = (level_map-(lvl-1)).clamp(min=0,max=1)
        retval = torch.lerp(retval,mipmap[lvl],blend)
#        from image_utils import plot_image
#        plot_image(blend,title=f'blend level {lvl}')
#        plot_image(retval,title=f'blurred level{lvl}')
#    plot_image(level_map,title='mipmap level map')
    return retval
    
# Note: using supersampling is now deprecated and will be removed eventually (adds extra expense and blurring, using preblur is better and faster)
# Note: preblur is still somewhat prmitive and could be improved with multiple levels, better blur kernel, etc.
def _gaze_warp_image_logpolar(image, params, gaze=None, 
                    mode='bicubic',rmax=None,preblur=True,make_mask=True):    
    width = image.size(-1)
    height = image.size(-2)
    if gaze is None: gaze = (height/2,width/2)
    eccentricity_scaling = params.scaling
    target_pooling_size = params.target_pooling_size
    anisotropy_ratio = params.anisotropy
    rmin = params.get_min_radius()
    # calculate # of distinct angles to use in the polar space
#    if params.round_azimuth:  # force the azimuthal rsesolution to be a multiple of the pooling size (important to avoid possible artifacts at 0/360 boundary)
#        thetasize = target_pooling_size*round(anisotropy_ratio*2*math.pi / eccentricity_scaling + 0.5)
    if params.azimuth_multiple: # ensure azimuth size is a multiple of the specified value (typically the pooling stride)
        thetasize = _ceil_multiple(target_pooling_size*anisotropy_ratio*2*math.pi / eccentricity_scaling, params.azimuth_multiple)
    else:
        thetasize = _ceil_multiple(target_pooling_size*anisotropy_ratio*2*math.pi / eccentricity_scaling, 8)
    
#    if rmin is None: rmin = target_pooling_size/eccentricity_scaling - target_pooling_size/2
    if rmax is None: rmax = _max_radius(image, gaze)
    a = target_pooling_size/eccentricity_scaling
    b = -a*math.log(rmin)
    smin = 0
    smax = _ceil_multiple(a*math.log(rmax) + b, 8)
#    print(f'thetasize {thetasize}  rmin{rmin} rmax{rmax} smin{smin} smax{smax}')
    # start with a uniform spacing in theta
    thalfstep = math.pi/thetasize
    theta = torch.linspace(-math.pi+thalfstep,math.pi-thalfstep,thetasize,device=image.device)
    # and convert to cos and sin
    cost = torch.cos(theta)
    sint = torch.sin(theta)
    # start with a uniform spacing in s
    simg = torch.arange(smin,smax,device=image.device).flip(-1)
    # convert to the radius values that would generate these s value
    rimg = torch.exp((simg-b)*(1/a))
    x = torch.outer(rimg,cost) + (gaze[1] + 0.5)    # grid sample uses coordinates where pixel centers are at the half positions
    y = torch.outer(rimg,sint) + (gaze[0] + 0.5)
    
    # convert to grid_samples' normalized coordinates
    x = (2/(width))*x - 1
    y = (2/(height))*y - 1
    grid = torch.stack((x,y),2).unsqueeze(0)
    if params.gamma:
        image = image.pow(params.gamma)  # Convert image to "linear" space if it was in a nonlinear gamma encoding 
    source = image
    # compute preblur if needed (used to reduce aliasing in regions where the warp is compressive)
    if preblur:
        radius,dx,dy = _make_distance_image(image,gaze,device=image.device)
        compression = radius/a
        source = _preblur_image_mipmap(image, compression.log2())

    # use grid_sample to fetch each pixel's value from its warped position
    retval = torch.nn.functional.grid_sample(source,grid,align_corners=False,mode=mode,padding_mode='border')
    
    #create mask of unneeded pixels outside of warped image's region
    mask = None
    if make_mask:
        mask = torch.nn.functional.grid_sample(torch.ones_like(source),grid,align_corners=False,mode=mode)
        mask = F.max_pool2d(mask,3,stride=1,padding=1) # Expand support region by 1 pixel in all directions (needed to get edge pixels fully covered during unwarp)
        mask = mask<0.0001
        retval[mask] = 0  # force pixels outside of image's warped support to zero
    if params.clamp_range:
        retval = torch.clamp(retval,min=0,max=1)
    if params.gamma:
        retval = retval.pow(1/params.gamma)  # convert back to same gamma encoding as the input used
    return retval,mask

def _gaze_unwarp_image_logpolar(warped,origsize,params,gaze=None,
                    mode='bicubic',rmax=None):
    if torch.is_tensor(origsize): origsize = origsize.size()
    width = origsize[-1]
    height = origsize[-2]
    thetasize = warped.size(-1)
    if gaze is None: gaze = (height/2,width/2)
    eccentricity_scaling = params.scaling
    target_pooling_size = params.target_pooling_size
    anisotropy_ratio = params.anisotropy
    rmin = params.get_min_radius()
#    if rmin is None: rmin = target_pooling_size/eccentricity_scaling - target_pooling_size/2
    if rmax is None: rmax = _max_radius(origsize, gaze)
    a = target_pooling_size/eccentricity_scaling
    b = -a*math.log(rmin)
    smin = 0
    smax = _ceil_multiple(a*math.log(rmax) + b, 8)
    if smax != warped.size(-2): raise ValueError(f' size mismatch {smax} vs {warped.size(-2)}')

    radius,dx,dy = _make_distance_image(origsize,gaze,device=warped.device)
    theta = (torch.atan2(dy,dx) + math.pi)*(thetasize/(2*math.pi))
    s = a*torch.log(radius) + b + 0.5
    nx = (2/(thetasize))*theta - 1
    ny = 1 - (2/smax)*s
    source = warped
    if mode != 'nearest':
        source = torch.nn.functional.pad(source,[2,2,0,0],'circular') #add values just beyond theta=-pi and theta=pi for interpolation
        nx = (thetasize/(thetasize+4))*nx
    grid = torch.stack((nx,ny),2).unsqueeze(0)
    if params.gamma:
        source = source.pow(params.gamma)  # Convert image to "linear" space if it was in a nonlinear gamma encoding 
    # use grid_sample to fetch each pixel's value from its warped position
    retval = torch.nn.functional.grid_sample(source,grid,align_corners=False,mode=mode,padding_mode='border')
    if params.clamp_range:
        retval = torch.clamp(retval,min=0,max=1)
    if params.gamma:
        retval = retval.pow(1/params.gamma)  # convert back to same gamma encoding as the input used
    return retval


# Warp an image into gaze-centric log-polar space
def gaze_warp_image(image,params,gaze=None,make_mask=True,prefer_cuda=True):
    if hasattr(gaze,"get_in_torch_coordinates"):  # automatically convert ImagePoint or NormalizeImagePoint classes to simple tuple
        gaze = gaze.get_in_torch_coordinates(image)
    # should we move the tensor to the GPU?
    move_to_cuda = prefer_cuda and torch.cuda.is_available() and not image.is_cuda
    if move_to_cuda: image = image.cuda()
    warped,mask = _gaze_warp_image_logpolar(image,params,gaze,make_mask=make_mask)
    if move_to_cuda:   # Move results back to cpu if that is where the origal image ws
        warped = warped.cpu()
        if mask is not None: mask = mask.cpu()
#    plot_image(warped,title='warped')
    if make_mask:
        return warped,mask
    else:
        return warped

# Unwarp log-polar image back to normal image space (inverse of gaze_warp_image, though there is some loss of information/blurring)
def gaze_unwarp_image(warpedimage,origimage,params,gaze=None,blend_gaze=True,draw_gaze=False,prefer_cuda=True):
    if hasattr(gaze,"get_in_torch_coordinates"):  # automatically convert ImagePoint or NormalizeImagePoint classes to simple tuple
        gaze = gaze.get_in_torch_coordinates(origimage)
        
    # Should we move the image to the GPU if it is not already there?
    move_to_cuda = prefer_cuda and torch.cuda.is_available() and not warpedimage.is_cuda
    if move_to_cuda: warpedimage = warpedimage.cuda()
    unwarped = _gaze_unwarp_image_logpolar(warpedimage,origimage.size(),params,gaze)
    if move_to_cuda: unwarped = unwarped.cpu()  # TODO: delay move back to cpu until after blends
    
    if blend_gaze:
        # The region around the gaze point where radius < min_radius will be wrong since it was not represented in the warped image
        # We can optionally blend in this region from the original image to fill in this missing region
        rmin = params.get_min_radius()
        kern = blend.trapezoid_radial(origimage,rmin+1,rmin+5,center=gaze)
        unwarped = blend.blend_images(origimage,unwarped,kern)
    if draw_gaze:
        # Can optionally draw a red dot at the gaze point
        dotcolor = torch.tensor([1.,0,0])[None,:,None,None]  
        dot_radius = 5
        if isinstance(draw_gaze,float): dot_radius = draw_gaze
        dotkern = blend.trigezoid_radial(unwarped,dot_radius,dot_radius+4,center=gaze)
        unwarped = blend.blend_images(dotcolor,unwarped,dotkern)
    return unwarped

def test_warp_roundtrip():
    from image_utils import plot_image, load_image_rgb
    start = time.time()
    # params
#    params = WarpParams(0.46,128,anisotropy=1)
#    params = WarpParams(0.9,128,anisotropy=1)
    params = WarpParams(1.9,128,anisotropy=1)
    gaze = None  #None will use default which is the center of image
    
    siz = 128
    image = torch.zeros(1,1,siz,siz)
    image[0,0,siz//4,siz//2] = 1
    image[0,0,siz//8,siz//2] = 1
    image[0,0,siz//2,siz//4] = 1
    image[0,0,siz//2,siz//16] = 1
    image[0,0,5*siz//6,siz//2] = 1
    image[0,0,siz//2,2*siz//3] = 1
    
    gaze = None
#    gaze = (32,132)
    image = load_image_rgb('../sampleimages/cat256.png')
#    ptdir = os.path.expanduser('../../data/papervideostests/')
#    deskdir = os.path.expanduser("~/Desktop/")
#    image = load_image_rgb(ptdir+"tokyocrossing_original.png")
    
    warped,mask = gaze_warp_image(image, params,gaze)
    unwarped = gaze_unwarp_image(warped, image, params, gaze)
    
    print(f'Elapsed time {time.time()-start} secs')
    plot_image(warped,title='warped')
    
    plot_image(image,title='original')
    plot_image(unwarped,title='after')
    plot_image((image-unwarped).abs(),title='diff')

    
if __name__ == "__main__":
    test_warp_roundtrip()
    
    