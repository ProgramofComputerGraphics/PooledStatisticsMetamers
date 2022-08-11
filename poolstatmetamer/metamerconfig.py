#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:42:10 2021

MetamerConfig is intended to make it easier to create metamers (ie images with
matching pooled statistics) by bundling together a variety of optional settings
with routines to create and configure the needed objects and steps.  MetamerConfig
uses MetamerImageSolver to compute the metamer images, but provides a simpler 
interface to configure and initialize the solvers, makes it easier to setup
parametric tests where you vary some parameters, and can generate video
sequences as well as still images.


The configuration parameters are passed to the constructor of MetamerConfig. 
A list of MetamerConfig objects can then be used to create images using these
preconfigured settings.  For example to generate metamers with varying pooling size,
or to setup a parametric sequence where you vary some metamer generation parameters.

See the samplescripts directory for some examples of how MetamerConfig can used

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import torch
import os
import math
import time
import re
import gc
import poolingregions as pool
import spyramid as sp
import imageblends as blend
from metamersolver import make_solver, MetamerImage
from gazewarp import gaze_warp_image, gaze_unwarp_image, WarpParams

from image_utils import load_image_gray, load_image_rgb, plot_image, LoadMovie, LoadMovieGray, LoadFrames, show_image, save_image

#---------- Some simple methods for generating seed images for the metamer solver-------------------
def seed_random(target,metamer_prior_frames=None,backup=None):
    return torch.rand_like(target)  # Seed with tensor of random numbers in range [0,1)

def seed_prior_frame(target,metamer_prior_frames=None,backup=None):
    if (metamer_prior_frames is not None) and (len(metamer_prior_frames)>=1): 
        return metamer_prior_frames[0]            # return previous metamer image as the seed if available
    if backup is not None: return backup(target)  # next try the backup seed method if available
    return torch.rand_like(target)                # otherwise use a random noise image

def seed_const_half(target,metamer_prior_frames=None,backup=None):
    return 0.5*torch.ones_like(target)   # Use a constant gray image as the seed

def seed_const_zero(target,metamer_prior_frames=None,backup=None):
    return torch.zeros_like(target)   # Use a constant black image as the seed

def seed_rotate180(target,metamer_prior_frames=None,backup=None):
    return torch.flip(target,(-1,-2))  #flip target image horizontally and vertically (equivalent to 180 rotation)

def seed_copy_target(target,metamer_prior_frames=None,backup=None):
    return target  #copies original image

# MetamerConfig class wraps the metamersolver interface to make it more easily configurable while
# also integrating some ease-of-use and sanity-check improvements.  You can instantiate MetamerConfig or
# lists of MetamerConfigs and apply them to images or video sequences.
class MetamerConfig():
    
    # This is the default directory where output files (metamer images, convergence graphs, movies, etc) will be stored
    DEFAULT_OUTPUT_DIR = './pmetamer_output'     # This is a class variable that affects all MetamerConfig instances
    # You can use this class method to change the default output directory for all usages of MetamerConfig
    @classmethod
    def set_default_output_dir(cls,outdir): cls.DEFAULT_OUTPUT_DIR = outdir
    
    def __init__(self,suffix='',
                 stats='',
                 copy_original=False,
                 pyramid=None, 
                 pooling=None, 
                 image_seed=seed_random, randseed = 49832475, 
#                 images_prefilter=None,
                 temporal_mode=None,
                 warping=None,
                 stat_modes=None,
                 solver_kwargs=None, solver_modes=None):
        super().__init__()
        self.suffix = suffix
        self.stat_params = stats
        self.copy_original_exactly = copy_original
        self.pyramid_params = sp.SPyramidParams.normalize(pyramid)
        self.pooling_params = pool.PoolingParams.normalize(pooling)
#        self.images_prefilter = images_prefilter
        if temporal_mode is not None: self.solver_kwargs['temporal_mode'] = temporal_mode
        self.randseed = randseed
        self.default_metamer_seed = image_seed
        self.warp_params = WarpParams.normalize(warping)
        self.solver_modes = {       # Default mode settings for solver, can be modified by subclasses
                'print_num_statistics':True,
                'print_image_comparison':True,
                'print_convergence_graph':True,
                'save_convergence_graph':True,
                'use_gpu_if_available':True
        }
        if solver_modes is not None: self.solver_modes.update(solver_modes)
        self.solver_kwargs = {}
        if solver_kwargs is not None: self.solver_kwargs.update(solver_kwargs)
        self.pooling_kwargs = {}    # Extra arguments to be passed to pooling generation
        self.stat_modes = {}        # Extra options to be set/changed in the statistics evaluation
        if stat_modes is not None: self.stat_modes.update(stat_modes)
        self.max_prior_frames_used = None       # Will be set automatically when solver is created
                
    # Combines explicit option with configuration settings to get the PoolingParams object to use
    def _get_pooling_params(self,pooling_sizes):
        if self.pooling_params is not None:
            pooling = self.pooling_params
            if self.pooling_kwargs: raise ValueError('cannot combine old and new pooling specifications')
            if (pooling_sizes is not None) and (pooling.get_width() is not None) and (pooling_sizes != pooling.get_width()):
                raise ValueError(f'pooling_size must be set to None if being overridden by a MetamerConfig schedule {pooling_sizes} vs {pooling.get_width()}')            
            if pooling_sizes is not None: pooling.set_width(pooling_sizes)
            if pooling.get_width() is None: raise ValueError(f'Must set the pooling size in either schedule or function call, both are current set to None: {self.pooling_params} {pooling_sizes}')
        else:
            pooling = pool.PoolingParams.normalize(pooling_sizes)
        return pooling
        
    # Override this to customize the pooling region construction
    def _create_pooling(self,pooling_sizes,target_image,gaze_point):
        if self.copy_original_exactly: return (pool.PoolingParams(math.inf),None)  # pooling doesn't matter if we are copying exactly
        if self.pooling_params is not None:
            pooling = self._get_pooling_params(pooling_sizes)
            copymask = None
        else:
            # Create the pooling-regions and copy-pixels-mask    
            if self.warp_params is not None:
                eccentricity_scaling = self.warp_params.scaling
            else:
                eccentricity_scaling = None
            (pooling,copymask) = pool.make_gaze_centric_pooling(pooling_sizes,target_image,gaze_point,eccentricity_scaling,**self.pooling_kwargs)
        return (pooling,copymask)
    
    # Override this to customize the solver or its configuration
    def _create_solver(self,target_image,pooling, outfile,outdir):
        solver = make_solver(target_image,pooling,pyramid_params=self.pyramid_params,**self.solver_kwargs)  
        # Turn on or off various informational outputs in the solver
        for mode,value in self.solver_modes.items():
            solver.set_mode(mode,value)
        solver.set_mode('save_image',outfile)
        solver.set_output_directory(outdir)
        self.max_prior_frames_used = solver.get_statistics_evaluator().max_prior_frames_used()
        
        # Once we have created the solver, we can recongfigure its statistics in various ways
        # Specifically we parse a string indicating statistics to enable '+stat_name' or disable '-stat_name'
        # and will disable all default statistics if the string doesn't start with an operator (+ or -)
        if self.stat_params:    
            stateval = solver.get_statistics_evaluator()
            # parse a string consisting of statistics to enable '+stat_name' or disable '-stat_name'
            slist = re.split(r'\s*([+-])\s*',self.stat_params.strip()) # split by + or - and remove whitespace around these operators
            cur = 0
            if slist[cur] != '':  #if string did not start with a + or - then disable all default statistics and then prepend a '+' to the lsit
                print('disabling all default statistics')
                stateval.set_all_stats(False)
                slist.insert(0,'+')
            else: 
                slist = slist[1:]  # remove leading blank token
            if len(slist) % 2 == 1:  # list should have even length (should consist of alternating operators and names)
                raise ValueError(f'Unable to parse stat string {self.stat_params}, unbalanced tokens {slist}')
            while cur < len(slist):
                if cur+1 >= len(slist): raise ValueError(f'Missing name field at end of statistics string {self.stat_params}')
                sname = slist[cur+1]
                if slist[cur] == '+':                
                    print(f'enabling stat: {sname}')
                    stateval.set_stat(sname,True)
                elif slist[cur] == '-':
                    print(f'disabling stat: {sname}')
                    stateval.set_stat(sname,False)
                else:
                    raise ValueError(f'Unrecognized operator {slist[cur]} in statistics string {self.stat_params}')
                cur = cur + 2
        if self.stat_modes:
            stateval = solver.get_statistics_evaluator()
            for mode,value in self.stat_modes.items():
                stateval.set_mode(mode,value)
        return solver
                            
    def update_namesuffix(self,prevsuffix):
        if self.suffix.startswith('&'):
            return prevsuffix+self.suffix[1:]
        else:
            return self.suffix
        
    # Constructs a metamer solver from some basic parameters, solves for the metamer, and saves the resulting image
    # Default is for a uniform metamer (with single pooling size), but you can also give a gaze point and a list
    # of pooling sizes to use.  Will include temporal statistics if the prior frames are supplied
    def generate_image_metamer(self,target_image,pooling_sizes=None,seed_image=None,max_iters=10,outfile='pmetamer.png',
                             outdir=None,randseed=None,
                             gaze_point=None,
                             target_prior_frames = (), metamer_prior_frames = ()):
        if outdir is None: outdir = self.DEFAULT_OUTPUT_DIR
        outpath = os.path.expanduser(outdir)  #expand ~ or ~user to the user's home directory
        # if a randseed was provided, use it 
        if randseed is None: randseed = self.randseed    # method parameter seed takes precedence otherwise use solver's seed
        if randseed is not None: torch.manual_seed(randseed)   # Setting the seed makes it (mostly) deterministic. Gpu scheduling can make it not fully deterministic
        # create seed image according to inputs (or use default method if not specified)
        if self.copy_original_exactly:  
            seed_image = target_image  #modify parameters to output will just be a copy of the original
        elif callable(seed_image): #if seed_image is a generator method then call it to generate the seed_image
            seed_image = seed_image(target_image,metamer_prior_frames=metamer_prior_frames,backup=self.default_metamer_seed)
        elif seed_image is None:   # otherwise use the solver's seed generation method
            seed_image = self.default_metamer_seed(target_image,metamer_prior_frames=metamer_prior_frames)
        if seed_image is target_image:
            max_iters = -1  # if seed and target image the same, no need for iterations
        # don't give solver previous metamer frames unless we are also giving it the previous target frames
        if len(target_prior_frames)==0: 
            metamer_prior_frames = ()  
        needs_unwarp = False
        copymask = False
        if self.warp_params is not None:
            if gaze_point is None: gaze_point = pool.NormalizedPoint(0.5,0.5)  #default is center of image if no gaze point supplied
            if sp.SPyramidParams.union(self.pyramid_params).boundary_mode.lower() != 'wrap_x':
                raise ValueError(f'Warped metamers should use wrap_x boundary mode for their pyramid, instead got {self.pyramid_params}')
            self.warp_params.suggest_pooling_params(self._get_pooling_params(pooling_sizes))
            orig_target = target_image
            target_image,warpcopymask = gaze_warp_image(target_image,self.warp_params,gaze_point)
            seed_image = gaze_warp_image(seed_image,self.warp_params,gaze_point,make_mask=False)
            needs_unwarp = True
            if warpcopymask is not None: copymask = warpcopymask | copymask
            #TODO: BW: should we set a small copy region here to help the metamer align with the foveal region for later blending????
        
        # Create the pooling-regions and copy-pixels-mask 
        (pooling,poolcopymask) = self._create_pooling(pooling_sizes,target_image,gaze_point)
        if poolcopymask is not None: copymask = poolcopymask | copymask
        # Create the metamer solver
        solver = self._create_solver(target_image,pooling, outfile,outdir=outpath) 
            
        # Compute our own metamer starting from precomputed metamer
        res = solver.solve_for_metamer(target_image,max_iters,seed_image,copy_target_mask=copymask,target_prior_frames=target_prior_frames,metamer_prior_frames=metamer_prior_frames)
        if needs_unwarp:
            unwarped = gaze_unwarp_image(res.get_image(),orig_target,self.warp_params,gaze_point)
            plot_image(target_image,title='warped original')
            plot_image(res.get_image(),title='warped metamer')
            plot_image(unwarped,title='unwarped metamer')
            res = MetamerImage(unwarped)
            # TODO: the following is a bit of hack to store unwarped output, should better integrate warping into metamersolver somehow
            if outfile is not None:
                savefile = os.path.join(outpath,outfile)
                os.replace(savefile,savefile+'.warped.png')
                if max_iters >= 0:
                    save_image(unwarped,savefile)
                else:
                    save_image(unwarped,savefile+'.unwarped.png')
                    save_image(orig_target,savefile)
            #Should we wrap this in a MetamerImage object?  Provide access to the warped version?
        return res
    
    # generate metamer of a movie (specified as a list of source frames)
    def generate_movie_metamer(self,source_generator,pooling_size,max_iters=500,gaze_point=None,
                   outbasename='pmetamer', outdir=None,
                   framerate=None, target_modifier=None,
                   use_prior_as_seed=True):
        # create directory and path where we will put the output
        if outdir is None: outdir = self.DEFAULT_OUTPUT_DIR
        outpath = os.path.expanduser(outdir)  #expand ~ or ~user to the user's home directory
        if outbasename is not None:
            outpath = os.path.join(outpath,f'{outbasename}movie{int(time.time())}')
        
        # no previous frame for the first frame
        target_prior_frames = []
        metamer_prior_frames = []
        seed_image = None
        if use_prior_as_seed:
            seed_image = seed_prior_frame    #use prior frame as seed
        for framenum,target_image in enumerate(source_generator):
            outfile = f'{outbasename}_frame{framenum:03d}.png'
            if outbasename is None: outfile=None
            if target_modifier: target_image = target_modifier(target_image)
            if self.use_movie_warping:
                target_image_original = target_image
                target_image = gaze_warp_image(target_image)
                outfile_unwarped = outfile
                outfile=None
#            print(f"target {target_image.size()} prev {target_prev_image.size() if target_prev_image!=None else None}")
#            print(f"prior frames {len(target_prior_frames)}")
            # generate the metamer for this frame
            res = self.generate_image_metamer(target_image,pooling_size,seed_image=seed_image,max_iters=max_iters,
                                     gaze_point=gaze_point,outfile=outfile,outdir=outpath,
                                     target_prior_frames=target_prior_frames,metamer_prior_frames=metamer_prior_frames)
            # prepend new images to list of prior frames (and truncate list if needed)
            if self.max_prior_frames_used > 0:
                target_prior_frames = [target_image, *target_prior_frames[0:self.max_prior_frames_used-1]]
                metamer_prior_frames = [res.get_image(), *metamer_prior_frames[0:self.max_prior_frames_used-1]]
        
        if outbasename is not None:
            blend.compile_frames_to_mp4(inputpattern=f'{outpath}/{outbasename}_frame%03d.png',outputfilename=f'{outpath}/{outbasename}_movie.mp4',framerate=framerate)
        print('finished movie generation')
        
""#END-CLASS------------------------------------    

# Generate image metamers for a specified image according schedule of configurations
# The schedule can be a single MetamerConfig or a list of them
def generate_image_schedule(target,config_schedule,pooling_sizes=None,color=False,seed_image=None,max_iters=10,basename='pmetamer',
                             target_modifier=None, randseed=None, gaze_point=None):
    # Load a original image 
    if torch.is_tensor(target):
        target_image = target
    elif color:
        target_image = load_image_rgb(target)
        if isinstance(seed_image,str): seed_image = load_image_rgb(seed_image)
    else:
        target_image = load_image_gray(target)
#        plot_image(target_image,title='loaded')
        if isinstance(seed_image,str): seed_image = load_image_gray(seed_image)
    if target_modifier: target_image = target_modifier(target_image)
    outfile = None
    suffix = ''
    for config in config_schedule:
        gc.collect()   #We can use a lot of memory, try to make sure as much is free as possible before starting
        suffix = config.update_namesuffix(suffix)
        if basename is not None: outfile = basename+suffix+".png"
        config.generate_image_metamer(target_image,pooling_sizes,seed_image,max_iters=max_iters,outfile=outfile,gaze_point=gaze_point)
    print('finished metamer generation schedule')
    
# generate metamer of a movie (specified as a list of source frames)
def generate_movie_schedule(source_generator,pooling_size,config_schedule,max_iters=500,gaze_point=None,
                   outbasename='pmetamer', framerate=None, target_modifier=None,
                   use_prior_as_seed=True, use_warping=False):

    suffix = ''
    print(f'item in sched {len(config_schedule)}')
    for config in config_schedule:
        gc.collect()   #We can use a lot of memory, try to make sure as much is free as possible before starting
        suffix = config.update_namesuffix(suffix)
        baseimagename = None
        if outbasename is not None: baseimagename = outbasename+suffix
        config.generate_movie_metamer(source_generator,pooling_size,
                                      max_iters=max_iters,gaze_point=gaze_point,
                                      outbasename=baseimagename,framerate=framerate,target_modifier=target_modifier,
                                      use_prior_as_seed=use_prior_as_seed)
    print('finished movie generation schedule')


def _test_metamer_config():

    # Portilla&Simoncelli-style steerable pyramid (1 highpass, 4 bandpass-edge, and 5 lowpass levels with 4 orientations and using cosine for radial high/low kernels)
    # additionally it will treat the image boundaries as wrapping around (torus topology)
    PS_pyr = 'UBbbbL_6:Ori=4:RadK=cos:Bound=wrap'
    # Portilla&Simoncelli-style pooling where there is only a single poolnig region and it covers the entire image
    PS_pool = 'whole' 

    # Freeman&Simoncell-style statistics and pooling (gaze-centric through use of log-polar warp) 
    FS_gaze_warp = "warp=0.75:anisotropy=2"
    FS_gaze_pyr = "UBbbbbbL_8_:Ori=4:Bound=wrap_x"
    FS_gaze_pool = '96:Kern=Trig:mesa=1/3:stride=2/3:Bound=wrap_x'

    # A few simple shcedules as examples
    poolsize = 128
    TestSched = (
        MetamerConfig('_original',copy_original=True),  # Will just copy the target image (sometimes useful to record what it was)
        MetamerConfig('_P&S',pooling=PS_pool, pyramid=PS_pyr, stats='ps_all'),  # Make Portilla&Simoncelli-style metamer with single pooling region and their statistics
        MetamerConfig('_meanonly',pooling=poolsize,stats="mean",image_seed=seed_const_half),  # Use only the mean (per pooling region) statistic
        MetamerConfig('_F&Sgaze',pooling=FS_gaze_pool, pyramid=FS_gaze_pyr, warping=FS_gaze_warp, stats='fs_all')
        #    MetamerConfig('_pmet128', pooling=128, pyramid="UEeeeee_7_:Ori=6", solver_modes={'save_convergence_movie':True}),    
    )
    iters = 16 #note this is just for testing, you typically need hundreds of iterations to achieve reasonable convergence
    generate_image_schedule('../sampleimages/cat256.png',TestSched,color=False,max_iters=iters,basename=None)

if __name__ == "__main__":   # execute main() only if run as a script
    _test_metamer_config()
