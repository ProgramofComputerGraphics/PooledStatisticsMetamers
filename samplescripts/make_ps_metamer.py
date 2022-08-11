# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:03:58 2021

Example code generating Portilla&Simoncelli style texture synthesis.  The pooled
statistics metamers build upon the P&S statistics and analysis, so this is useful
as a basic test of some of the basic methods of the PooledStatisticsMetamers project.

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import sys
sys.path.append('../poolstatmetamer')  # Hack to allow importing from poolstatmetamer sibling package/directory
from metamerconfig import MetamerConfig, generate_image_schedule

# Change this to wherever you want the output files to be saved
MetamerConfig.set_default_output_dir('./poolstatmetamer_output')  # default directory where output files will be stored

# Portilla&Simoncelli-style steerable pyramid (1 highpass, 4 bandpass-edge, and 5 lowpass levels with 4 orientations and using cosine for radial high/low kernels)
# additionally it will treat the image boundaries as wrapping around (torus topology)
PS_pyr = 'UBbbbL_6:Ori=4:RadK=cos:Bound=wrap'   
# Portilla&Simoncelli-style pooling where there is only a single poolnig region and it covers the entire image
PS_pool = 'whole' 

# A schedule is a list of metamer settings to use to create images
PS_Sched = (
        MetamerConfig('_original',copy_original=True),  # Will just copy the target image (sometimes useful to record what it was)
        MetamerConfig('_P&S',pooling=PS_pool, pyramid=PS_pyr, stats='ps_all'),  # Make Portilla&Simoncelli-style metamer with single pooling region and their statistics
)

#iters = 16 #note this is just for testing, you typically need hundreds of iterations to achieve convergence in many cases
iters = 300

# Generate an Portilla&Simoncelli-style metamer for the pluses image
generate_image_schedule('../sampleimages/pluses.png',PS_Sched,color=False,max_iters=iters,basename='plus')
