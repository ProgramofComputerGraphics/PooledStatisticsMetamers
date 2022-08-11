#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:46:28 2021

Example code for generating a gaze-centric pooled statistics metamer.
The pooling regions grow linearly in size with distance with the gaze point
(ie eccentricity).  Internally this is accopmlished by transforming the image
into a log-polar apace to equality the pooling region sizes, generating a uniform
metamer in this space and then tranforming the result back into normal image space.

Parameters such as the scale of the warp, the type of pooling regions used, the
set of image statistics, etc, can be configured using the various parameter
strings.

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import sys
sys.path.append('../poolstatmetamer')  # Hack to allow importing from poolstatmetamer sibling package/directory
from metamerconfig import MetamerConfig, generate_image_schedule, seed_const_half

# Change this to wherever you want the output files to be saved
MetamerConfig.set_default_output_dir('./poolstatmetamer_output')  # default directory where output files will be stored

# Portilla&Simoncelli-style steerable pyramid (1 highpass, 4 bandpass-edge, and 5 lowpass levels with 4 orientations and using cosine for radial high/low kernels)
# additionally it will treat the image boundaries as wrapping around (torus topology)
PS_pyr = 'UBbbbL_6:Ori=4:RadK=cos:Bound=wrap'   
# Portilla&Simoncelli-style pooling where there is only a single poolnig region and it covers the entire image

# Pooling based on Freeman&Simoncelli's gaze-centric pooling
FS_gaze_pool = '96:Kern=Trig:mesa=1/3:stride=2/3:Bound=wrap_x'
# Pyramid based on Freeman&Simoncelli's settings but uses more levels to account for dynamic resizing that happens when using our warping transform
FS_gaze_pyr = "UBbbbbbL_8_:Ori=4:Bound=wrap_x"
# Warp settings based on Freeman&Simoncelli's gaze-centric eccentricity scaling of 0.46
# Note that although F&S used a eccentricity scaling of 0.46, they measured their the size of their pooling kernels differently 
# than we do (they used full-width-at-half-maximum, while we use size of the support).  So we need to adjust the eccentricity
# scaling to account for this difference.  For their pooling kernels our size is 1.5x larger so we need to set the scaling
# to be 0.5*1.5 = 0.75 to get similarly sized pooling regions as a function of eccentricity
# (or for example, if we to match their scaling value of 0.46, we would use: 0.46*1.5 = 0.69)
FS_gaze_warp = "warp=0.75:anisotropy=2"

# Pooling setting for gaze-centric metamer with high overlap (quarter-region spacing between region centers)
gaze_over4_pool = '96:Kern=Trig:mesa=1/2:stride=1/4:Bound=wrap_x'
# Pooling setting for gaze-centric metamer with medium overlap (half-region spacing between region centers)
gaze_over2_pool = '96:Kern=Trig:mesa=0:stride=1/2:Bound=wrap_x'
# Our current steerable pyramid settings for a gaze-centric image
better_gaze_pyr = "UEeeeee_7_:Ori=6:RadK=gauss:Bound=wrap_x"

# A schedule is a set of metamer configurations you want to generate for each image/movie
# You can add/remove/comment-out entries for the particular types of metamers you want to generate
FreemanWarpSched = ( # This schedule generate Freeman&Simoncelli-style gaze-centric metamers
    MetamerConfig('_original',copy_original=True, pooling=FS_gaze_pool),
    MetamerConfig('_fs_meanonly', pooling=FS_gaze_pool, pyramid=FS_gaze_pyr, warping=FS_gaze_warp, image_seed=seed_const_half, stats='mean'),
    MetamerConfig('_fs', pooling=FS_gaze_pool, pyramid=FS_gaze_pyr, warping=FS_gaze_warp, stats='fs_all' ),
    )

OurGazeWarpSched = (
#    MetamerConfig('_original',copy_original=True, pooling=gaze_over4_pool),
    MetamerConfig('_gazemet', pooling=gaze_over4_pool, pyramid=better_gaze_pyr, warping=FS_gaze_warp),
    )

#iters = 16 #note this is just for testing, you typically need hundreds of iterations to achieve convergence in many cases
iters = 300

# The generate_image_schedule command will generate metamers according to the settings in the specified schedule of MetamerConfig's
#generate_image_schedule('../sampleimages/cat256.png',FreemanWarpSched,color=True,max_iters=iters,basename='cat')

generate_image_schedule('../sampleimages/cat256.png',OurGazeWarpSched,color=True,max_iters=iters,basename='cat')
