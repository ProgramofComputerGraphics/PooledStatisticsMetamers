# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:01:45 2021

Example code for generating a uniform pooled statistics metamer.  The program will
compute a set of statistics for the target image, pool (ie average) them over a set
of uniform-sized overlapping regions.  Then starting from a seed image (usually random)
it will progressively modify the result image so that its pooled statistics more
closely match those of the target image.  The optimization method is gradient
descent, and it will typically require hundreds of iterations (or more) to produce
a good match.

This program can be run as is as a test of the system and can be used as a template
and modified to compute metamers of image and other types of metamers.  The process
can be configured by modifying the various parameter strings to change the statistics
used, the pooling regions, etc.

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import sys
sys.path.append('../poolstatmetamer')  # Hack to allow importing from poolstatmetamer sibling package/directory
from metamerconfig import MetamerConfig, generate_image_schedule

# Change this to wherever you want the output files to be saved
MetamerConfig.set_default_output_dir('./poolstatmetamer_output')  # default directory where output files will be stored


# Pooling setting for uniform metamer with high overlap (quarter-region spacing between region centers)
overlap4_pool = ':Kern=Trig:mesa=1/2:stride=1/4'
# Pooling setting for uniform metamer with medium overlap (half-region spacing between region centers)
overlap2_pool = ':Kern=Trig:mesa=0:stride=1/2'
# Pooling setting for uniform metamer with lower overlap based on the pooling overlap used by Freeman&Simoncelli (though they used gaze-centric metamers)
overlapFS_pool = ':Kern=Trig:mesa=1/3:stride=2/3'

# Steerable pyramid using settings used in many prior methods including Freeman&Simnoncelli 
FS_pyr = 'UBbbbL_6:Ori=4:RadK=cos'   

# Our current steerable pyramid settings for a gaze-centric image
gauss_pyr = "UEeeeee_7_:Ori=6:RadK=gauss"


# A schedule is a list of metamer settings to use to create images
# This schedule generates a uniform metamer where the pooling size specified in the generate_image_schedule() call
Uniform_Sched = (
        MetamerConfig('_original',copy_original=True),  # Will just copy the target image (sometimes useful to record what it was)
#        MetamerConfig('_unif',pooling=overlap4_pool, pyramid=gauss_pyr),  # Make metamer with uniform pooling regions with a high degree of overlap
        MetamerConfig('_unif',pooling=overlap4_pool, pyramid=gauss_pyr, solver_modes={'save_convergence_movie':True}),  # Same as above but also save convergence movie
                
)

# This schedule generates several uniform metamers with different pooling sizes (as specified here)
Uniform_Sizes_Sched = (
        MetamerConfig('_original',copy_original=True),  # Will just copy the target image (sometimes useful to record what it was)
        MetamerConfig('_unif48', pooling='48'+overlap4_pool,  pyramid=gauss_pyr),
        MetamerConfig('_unif96', pooling='96'+overlap4_pool,  pyramid=gauss_pyr),  
        MetamerConfig('_unif144',pooling='144'+overlap4_pool, pyramid=gauss_pyr),  
    )

# This schedule generates several uniform metamers with different densities of pooling regions (varying amounts of overlap) but using our statistics and steerable pyramid
Uniform_Overlap_Sched = (
        MetamerConfig('_original',copy_original=True),  # Will just copy the target image (sometimes useful to record what it was)
        MetamerConfig('_unif96_overlap4', pooling='96'+overlap4_pool,  pyramid=gauss_pyr),
        MetamerConfig('_unif96_overlap2', pooling='96'+overlap2_pool,  pyramid=gauss_pyr),  
        MetamerConfig('_unif96_overlapFS',pooling='96'+overlapFS_pool, pyramid=gauss_pyr),  
    )

# This schedule generates uniform metamers based on our current settings as well as some based on prior methods
Uniform_Stats_Sched = (
        MetamerConfig('_unif96_ours', pooling='96'+overlap2_pool,  pyramid=gauss_pyr),  
        MetamerConfig('_unif96_FS', pooling='96'+overlapFS_pool,  pyramid=FS_pyr, stats='fs_all'),  
    )

#iters = 16 #note this is just for testing, you typically need hundreds of iterations to achieve convergence in many cases
iters = 300

# Generate a uniform metamer for the cat image with pooling region size set to 128
generate_image_schedule('../sampleimages/cat256.png',Uniform_Sched,128,color=True,max_iters=iters,basename='cat')

# Generate uniform metamers for cat image with varying pooling sizes set in the schedule
#generate_image_schedule('../sampleimages/cat256.png',Uniform_Sizes_Sched,color=True,max_iters=iters,basename='cat')

# Generate grayscale uniform metamers for cat image using our default settings and setting's based on prior methods
#generate_image_schedule('../sampleimages/cat256.png',Uniform_Stats_Sched,color=False,max_iters=iters,basename='catg')

