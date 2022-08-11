#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:55:52 2019

This object contains code for computing various types of pooled statistics
images from source images.  The set of statistics can be configured and information
about the current statistics being used can be returned as a list of StatLabels

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

# Classes for computing statistics on (extended) steerable pyramids 
# mainly consisting of moments, covariances, and autocorrelations
# (such as those from Portilla&Simoncelli and Freeman&Simoncelli, as well as some new ones for color and time)
# These classes take prebuilt steerable pyrmaids and return lists of statistics computed on them

import torch
import autocorrelation as acorr
from spyramid import SPyramidParams
from typing import NamedTuple, Union, Tuple, Any

#imports for edgestop tests
import math
from autodifference import autodifference2d

# First some convenience generators for considering all pairs within a numbered list or between two equal sized lists

# Generator unordered pairs of two numbers >= 0 and < limit
# Ordering of elements does not matter (so only one of (a,b) and (b,a) is included)
# and that the pair elements must be distinct (so (a,a) will not be included)
# Useful for generating all unique pairs of elements from a single list includes limit*(limit-1)/2 pairs
def range_unique_pairs(limit):
    for i in range(limit-1):
        for j in range(i+1,limit):
            yield (i,j)
            
# Generator for all possible (ordered) pairs of numbers >= 0 and < limit  for limit^2 pairs
# Useful for generating all possible pairs between elements from two equal length lists
def range_all_ordered_pairs(limit):
    for i in range(limit):
        for j in range(limit):
            yield (i,j)

# Generator for all (ordered) pairs of numbers >=0 and < limit where the two elements must be different
# Similar to range_all_ordered_pairs(limit) but excludes pairs of the form (a,a)
def range_distinct_ordered_pairs(limit):
    for i in range(limit):
        for j in range(limit):
            if (i != j): yield (i,j)
    
# A record containing some information about a particular statistic mode 
# used to optionally return some information about each statistic for debugging or labeling outputs
class StatLabel(NamedTuple):
    weight_category: str          # Name for statistic's weighting category
    level: Union[int,Tuple[int]]  # Pyramid level(s) used for this statistic. We use -1 for base/original image 
    channel: Union[str,Tuple[str]]# Name for channel(s) used by this statistic (eg 'ac' for achromatic channel)
    temporal: Union[str,Tuple[str]]  # Names for temporal channels used by this statistic
    note: Any = None              # Any extra identifying information for this statistic
    orientation: Union[None,int,Tuple[int]] = None
                                  # Orientation(s) for this statistic
    
    def __str__(self):
        if self.channel is None:
            chstr = ''
        elif isinstance(self.channel,str):
            chstr = self.channel + ' '
        else:
            if len(self.channel) != 2: raise ValueError(self.channel)
            chstr = f'{self.channel[0]}_{self.channel[1]} '
        if self.level is None:
            lstr = ' Base'
        elif isinstance(self.level,int):
            lstr = f' L{self.level}'            
        else: 
            if len(self.level) != 2: raise ValueError(self.level)
            lstr = f' L{self.level[0]}_{self.level[1]}'
        if self.orientation is not None:
            if isinstance(self.orientation,int):
                lstr += f' Ori{self.orientation}'
            else:
                if len(self.orientation) != 2: raise ValueError(self.orientation)
                lstr += f' Ori{self.orientation[0]}_{self.orientation[1]}'
        if self.temporal: lstr += f' {self.temporal}'
        retval = chstr+self.weight_category+lstr
        if self.note is not None: retval += ' '+self.note
        return retval
""#END-CLASS------------------------------------
    
# Abstract superclass for various classes of metamer statistics
# A statistic mode is a set of one or more statistics which can be enable/disabled by setting the modes value,
# such as variances of the low-pass images or autocorrelations of edge magnitude images.
# Categories are used to weight the statistics.  This is used to correct for the fact that different
# image components have different typical ranges, but could also be used to emphasize or reduce the importance of some statistics
class MetamerStatistics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.stat_somename            # By convention boolean flags enabling various statistics have the prefix stat_
        self.per_level_weight = math.sqrt(2)   # weight statistics by per_level_weight^level (allows higher levels to get higher weights)
#        self.per_level_weight = None   # weight statistics by per_level_weight^level (allows higher levels to get higher weights)
        self._all_stats_list = None    # List of all statistic modes in this object (generated lazily from all fields starting with stat_)
        # subclasses can define a dictionary of named groups of statisitcs and store it here
        self.named_stat_groups = { }   # Empty dictionary here, subclasses can add entries
        #This set of weights was estimated to give each statistic similar variation using a very crude
        #estimation process.  Likely need to revisit this with more images, data, and experience
        #subclasses can modify or add categories
#        self.category_weights = { 'mean':6, 'variance':2, 'skew':2, 'kurtosis':2, 'high_variance':10, 
#                                 'autocorrelation':1, 
#                                 'edge_mean':20, 'edge_variance':80, 'edge_kurtosis':400, 
#                                 'edge_autocorrelation':5, 'edge_correlation':40, 
#                                 'phase_correlation':100}
        self.category_weights = { 'mean':2, 'variance':1, 'skew':1, 'kurtosis':1, 'bandpass_variance':40, 
                                 'autocorrelation':0.5, 
                                 'edge_mean':10, 'edge_variance':80, 'edge_kurtosis':800, 
                                 'edge_autocorrelation':2, 'edge_correlation':80, 'edge_stop':100, 'edge_continue':80,
                                 'phase_correlation':400,
                                 'covariance':1, 'edge_covariance':80, 
                                 'phase_covariance':400,
                                 }
        
    # Generic way to set modes (fields) that will raise an error if the given field does not already exist (detects spelling mistakes)    
    def set_mode(self,mode,value):
        if hasattr(self,mode):
            setattr(self,mode,value)
        elif mode in self.category_weights:
            self.category_weights[mode] = value
        else:
            raise NameError(f"Attribute or weight: {mode}  does not exist or is misspelled")
            
    # Turn on or off a particular statistic odes (note: name used here does not include stat_ prefix)
    # Returns true if the statistic exists (can optionally raise an exception if not found)
    def set_stat(self,attr,value, error_if_missing=True):
        attr = 'stat_'+attr       # Statistic modes use boolean field with stat_ prefix
        if hasattr(self,attr):
            setattr(self,attr,value)
            return True
        else:
            if error_if_missing: raise NameError(f"Attribute: {attr}  does not exist or is misspelled")
            return False
        
    # Turn on or off all the statistic modes implemented by this object            
    def set_all_stats(self,boolean):
        for s in self.get_list_of_stats():
            self.set_stat(s,boolean)
            
    # Get the enable/disable value for a particular mode (note: name used here does not include the stat_ prefix)
    def get_stat(self,attr):
        attr = 'stat_'+attr       # Statistic modes use boolean field with stat_ prefix
        if hasattr(self,attr):
            return getattr(self,attr)
        else:
            raise NameError(f"Attribute: {attr}  does not exist or is misspelled")
            
    def get_list_of_stats(self):
        if self._all_stats_list is None:
            # if it doesn't exist yet, autogenerate the list of all modes (from fields with stat_ prefix)
            stats = []
            for attr in self.__dict__.keys():
                if attr.startswith('stat_'):
                    attr = attr[5:]   #remove stat_ prefix
                    stats.append(attr)
            self._all_stats_list = stats
        return self._all_stats_list
    
    def set_stat_group(self,groupname,value):
        if groupname not in self.named_stat_groups: return False
        for stat in self.named_stat_groups[groupname]:
            if stat in self.named_stat_groups:
                self.set_stat_group(stat,value) # allow recursive group definitions
            else:
                self.set_stat(stat,value)
        return True
    
    def set_per_level_weight_multiplier(self,value):
        attr = 'per_level_weight'
        if hasattr(self,attr):
            setattr(self,attr,value)
        else:
            raise NameError(f"Attribute: {attr}  does not exist or is misspelled")
        
    def _level_weight(self,level):
        if level is None: return 1
#        if hasattr(level,'__iter__'): level = min(level)
        if hasattr(level,'__iter__'): level = max(level)  # Gives slightly higher weight for inter-level statistics
        return self.per_level_weight**level
        
    # Compute each of the enabled statistics, average it over the pooling regions, and return
    # the result as a list of tensor images (one per statistic)
    # If stat_labels is a list, then a StatLabel for each stat will be added to it
    # is_target should be set if image is target (one whose statistics we want to match)
    def forward(self,spyr,poolfunc, stat_labels=None, statlabel_callback=None):
        raise NotImplementedError('Subclasses must implement this method')
    
""#END-CLASS------------------------------------
    
# Simple class that does not evaluate any statistics (useful in cases where we
#   want the pyramids to be built for cross-correlations but don't need within-channel statistics)
class EmptyStatistics(MetamerStatistics):

    def __init__(self):
        super().__init__()

    def forward(self,spyr,poolfunc, stat_labels=None, statlabel_callback=None):
        return []  # just return an empty list as we have no statistics

""#END-CLASS------------------------------------
    
# Implements Portilla&Simoncelli style texture statistics (or Freeman&Simoncelli)
# The forward method takes an steerable pyramid and pooling function, and then computes its statistics, 
# pools them by region, and return the resulting statistics as a list of tensor images
class GrayscaleStatistics(MetamerStatistics):
    
    def __init__(self,autocorrelation_window=7):
        super().__init__()
        # List of offsets to use in autocorrelations (unlike the paper, we do not include (0,0), or variance here)
        self.low_autoshifts = acorr.generate_offset_list(autocorrelation_window)   
        self.edge_autoshifts = self.low_autoshifts
        # Classes of statistics to include (or not if set to false)
        self.stat_mean = True                         # We use mean of base image, but this also controls mean of the low-pass images
        self.stat_base_variance = True
        self.stat_base_skewkurtosis = False
#        self.stat_high_variance = True
        self.stat_bandpass_variance = True            # Generalization of old high_variance statistic    
#        self.stat_high_variance_to_zero = False       # Not actually a statistic, but option to try to force high-pass to zero in metamer
        self.stat_low_variance = True                 # Note we include variance here rather than in autocorrelation
        self.stat_low_skewkurtosis = True                
        self.stat_low_autocorrelation = False         # We turned off autocorrelations by default  bw:4/2020
        self.stat_edge_mean = True                    # Not in the Portilla&Simoncelli statistics but in Freeman's
        self.stat_edge_variance = True                # We include variance separately from autocorrelation
        self.stat_edge_kurtosis = False                # Not in Freeman's or Portilla&Simoncelli's statistics
        self.stat_edge_autocorrelation = False        # We turned off autocorrelations by default  bw:4/2020
        self.stat_edge_orientationcorrelation = True
        self.stat_edge_scalecorrelation = True       
        self.stat_edge_scaleorientationcorrelation = False # Separated from edge_scalecorrelation and turned off by default bw:8/2020
        # edge_stop is an experimental new statistic currently being tested
        self.stat_edge_stop = True
        self.stat_edge_continue = False   #experimental new statistics
        self.stat_phase_orientationcorrelation = True     # Not in Portilla statistics but in Freeman's (off by default bw:8/2020, renabled 12/2021, helps text tests)
        self.stat_phase_scalecorrelation = True
        self.stat_phase_scaleorientationcorrelation = False # Separated from phase_scalecorrelation and turned off by default bw:8/2020
        
        # Some named groups of statistics for convenience
        self.named_stat_groups = { 
                 # Portilla&Simoncelli categories
                 'ps_marginalstatistics':('mean','base_variance','base_skewkurtosis','low_skewkurtosis','bandpass_variance'),
                 'ps_coefficientcorrelation':('low_variance','low_autocorrelation'),
                 'ps_magnitudecorrelation':('edge_mean','edge_variance','edge_autocorrelation','edge_orientationcorrelation','edge_scalecorrelation','edge_scaleorientationcorrelation'), #note: original P&S did not use edge_mean, but we include it since we are using raw moments instead of centralized moments
                 'ps_crossscalephase':('phase_scalecorrelation','phase_scaleorientationcorrelation'),  
                 # Meta groups for all Portilla&Simoncelli statistics and all Freeman&Portilla statistics
                 'ps_all':('ps_marginalstatistics','ps_coefficientcorrelation','ps_magnitudecorrelation','ps_crossscalephase'),
                 'fs_all':('ps_all','phase_orientationcorrelation'),
                 # categories based on source image type (original,low-pass,high-pass,edge-magnitude,phase)
                 'base_stats':('mean','base_variance','base_skewkurtosis'),
                 'low_stats':('mean','low_variance','low_skewkurtosis','low_autocorrelation'),
                 'high_stats': ('bandpass_variance',),
                 'edge_stats':('edge_mean','edge_variance','edge_autocorrelation','edge_orientationcorrelation','edge_scalecorrelation','edge_scaleorientationcorrelation','edge_stop'),
                 'phase_stats': ('phase_orientationcorrelation','phase_scalecorrelation','phase_scaleorientationcorrelation'),
                # categories based on statistics type
                 'mean_stats':('mean','edge_mean'), 
                 'variance_stats':('base_variance','bandpass_variance','low_variance','edge_variance'),
                 'autocorrelation_stats':('low_autocorrelation','edge_autocorrelation'),
                 'skewkurtosis_stats':('base_skewkurtosis','low_skewkurtosis')
                 }
        
    def set_for_portilla_statistics(self):
        self.set_all_statistics(False)
        self.set_mode('ps_marginalstatistics',True)
        self.set_mode('ps_coefficientcorrelation',True)
        self.set_mode('ps_magnitudecorrelation',True)
        self.set_mode('ps_crossscalephase',True)
        
    def set_for_freeman_statistics(self):
        self.set_for_portilla_statistics()
        self.set_mode('phase_orientationcorrelation',True)
                    
    def set_autocorrelation_offsets(self,offsetlist):
        self.low_autoshifts = offsetlist
        self.edge_autoshifts = offsetlist
        
    # Compute each of the enabled statistics, average it over the pooling regions, and return
    # the result as a list of tensor images (one per statistic)
    # Can also record category and descriptive name for each statistic
    def forward(self,spyr,poolfunc,stat_labels,statlabel_callback=None):
        stats = []                                # List of statistics images to return
        # Base image is the image from which the steerable pyramid was built
        baseimg = spyr.original_image()
        params = spyr.params()
        basesize = baseimg.size()
        channel = spyr.cname
        temporal = spyr.tname
        # Utility function to process and add one statistic image to list
        # statimg is the statistics image and cat is its category
        # src1,src2 are used to optinoally create descriptive strings for each statistic
        def add_stat(statimg,catname,level,*,ori=None,note=None):
            #if len(stats)==5: plot_image(statimg,title=name)
            weight = self.category_weights[catname]
            if self.per_level_weight: weight *= self._level_weight(level)
            stat = poolfunc.pool_stats(statimg,basesize)
            stats.append(weight*stat)
            if stat_labels is not None: 
                stat_labels.append(StatLabel(catname, level, channel, temporal, note, ori))
                if statlabel_callback is not None:
                    statlabel_callback(stat,stat_labels[-1],statimg)
#                imgdir = os.path.expanduser('~/Desktop/statimagespep_es/')
#                plot_image(stat,title=str(stat_labels[-1]),savefile=imgdir+str(stat_labels[-1]).strip()+".png")
#                plot_image(stat,title=str(stat_labels[-1]))
        
        # Start adding the various image statistics
        if self.stat_mean:
            add_stat(baseimg,'mean',level=None)      # Mean of original/base image
        if self.stat_base_variance or self.stat_base_skewkurtosis:
            if self.stat_base_variance:
                add_stat(baseimg.pow(2),'variance',level=None)
            if self.stat_base_skewkurtosis:
                add_stat(baseimg.pow(3),'skew',level=None)
                add_stat(baseimg.pow(4),'kurtosis',level=None)
        # Add band-pass image statistics
        for i in params.bandpass_range():
            B = spyr.band_pass_image(i)
            if self.stat_bandpass_variance:
                add_stat(B.pow(2),'bandpass_variance',level=i)
        # Add low-pass image statistics
        for i in params.lowpass_range():
            L = spyr.low_pass_image(i)
            if self.stat_low_variance:
                add_stat(L.pow(2),'variance',level=i)
            if self.stat_low_skewkurtosis:
                add_stat(L.pow(3),'skew',level=i)
                add_stat(L.pow(4),'kurtosis',level=i)
            if self.stat_low_autocorrelation:
                # Scale increases with level but is decreased if image has been downsampled (reduced in size)
                scale = (2**i)*L.size(-1) // basesize[-1]
                #print(f"scale: {scale} origsize: {origsize} size: {L.size(-1)}")
                for shift in self.low_autoshifts:
                    add_stat(acorr.autocorrelation2d(L,shift,scale),'autocorrelation',level=i,note=shift)
        def _ori_to_list(x): # returns list of tensors for each slice in dimension one, but keeps the same number of dimensions (unlike torch.unbind)
            return [x.narrow(1,ori,1) for ori in range(x.size(1))] if x is not None else None
        # Add edge magnitude statistics
        for i in params.edge_range():
            Mlist = _ori_to_list(spyr.edge_magnitude_images(i))
            Clist = _ori_to_list(spyr.coarser_magnitude_images(i))  # may be None if coarser images are not present at this level
            if self.stat_edge_stop:      # Build list of offsets used by the edgestop statistics
                stopdist = 2**(i)
                edgestop_offsets = []
                for ori in range(len(Mlist)):
                    angle = ori*math.pi/(len(Mlist))
                    edgestop_offsets.append( (round(stopdist*math.cos(angle)),round(stopdist*math.sin(angle))) )
            if self.stat_edge_continue:      # Build list of offsets used by the edgecontinue statistics
                cdist = 2**(i+2)              # Note distance if four times greater than distance used for edge stop
                edge_continue_offsets = []
                for ori in range(len(Mlist)):
                    angle = ori*math.pi/(len(Mlist))
                    edge_continue_offsets.append( (round(cdist*math.cos(angle)),round(cdist*math.sin(angle))) )
            #Loop over the edge orientations 
            for ori,M in enumerate(Mlist):
                if self.stat_edge_mean:
                    add_stat(M,'edge_mean',level=i,ori=ori)
                if self.stat_edge_variance:         # We include variance here rather than in autocorrelation
                    add_stat(M.pow(2),'edge_variance',level=i,ori=ori)
                if self.stat_edge_kurtosis:
                    add_stat(M.pow(4),'edge_kurtosis',level=i,ori=ori)
                if self.stat_edge_autocorrelation:
                    scale = (2**i)*M.size(-1) // basesize[-1]
                    for shift in self.edge_autoshifts:
                        add_stat(acorr.autocorrelation2d(M,shift,scale),'edge_autocorrelation',level=i,ori=ori,note=shift)
                if self.stat_edge_scalecorrelation and Clist:
                    add_stat(M*Clist[ori], 'edge_correlation',level=(i,i+1),ori=ori)
                if self.stat_edge_stop:
                    add_stat(autodifference2d(M,edgestop_offsets[ori])**2,'edge_stop',level=i,ori=ori)
                if self.stat_edge_continue:
                    add_stat(acorr.autocorrelation2d(M,edge_continue_offsets[ori]),'edge_continue',level=i,ori=ori)
            if self.stat_edge_orientationcorrelation:   
                for (a,b) in range_unique_pairs(len(Mlist)):
                    add_stat(Mlist[a]*Mlist[b], 'edge_correlation',level=i,ori=(a,b))
            if self.stat_edge_scaleorientationcorrelation and Clist:
                for (a,b) in range_distinct_ordered_pairs(len(Mlist)):
                    add_stat(Mlist[a]*Clist[b], 'edge_correlation',level=(i,i+1),ori=(a,b))
                    
        # Add edge phase statistics
        for i in params.edge_range():
            er = _ori_to_list(spyr.edge_real_images(i))
            ei = _ori_to_list(spyr.edge_imag_images(i))
            #Note: to save some memory we can not build dr and only use the di images.  In this case dr==None even when di is present 
            dr = _ori_to_list(spyr.dphase_real_images(i))
            di = _ori_to_list(spyr.dphase_imag_images(i))
            if self.stat_phase_orientationcorrelation:
                for (a,b) in range_unique_pairs(len(er)):
                    add_stat(er[a]*er[b], 'phase_correlation',level=i,ori=(a,b),note='er')
            if self.stat_phase_scalecorrelation and di:
                for a in range(len(er)):
                    if dr is None:
                        add_stat(ei[a]*di[a], 'phase_correlation',level=(i,i+1),ori=a,note='ei*di')
                    else:
                        add_stat(er[a]*dr[a], 'phase_correlation',level=(i,i+1),ori=a,note='er*dr')
                    add_stat(er[a]*di[a], 'phase_correlation',level=(i,i+1),ori=a,note='er*di')
            if self.stat_phase_scaleorientationcorrelation and di:
                for (a,b) in range_distinct_ordered_pairs(len(er)):
                    if dr is None:
                        add_stat(ei[a]*di[a], 'phase_correlation',level=(i,i+1),ori=(a,b),note='ei*di')
                    else:
                        add_stat(er[a]*dr[b], 'phase_correlation',level=(i,i+1),ori=(a,b),note='er*dr')
                    add_stat(er[a]*di[b], 'phase_correlation',level=(i,i+1),ori=(a,b),note='er*di')
        #plot_image(stats[-1])
        return stats     # Return list of statistic tensors (assume loss function can process a list)
        
""#END-CLASS------------------------------------

# Cross-color-channel statistics (individual color channels can be handled by grayscale class)
# A somewhat reduced set of cross color channel statistics that just includes the covariance/correlation of
# the corresponding images in each channels steerable pyramid (lowpass,edge_magnitude,edge_phase)
# I did not inlcude the high-pass image because I suspect it can be safely ignored (contains little information)
# This somewhat reduced set needs testing and verification to see how it performs on a variety of color images
class CrossColorStatistics(MetamerStatistics):
    
    def __init__(self):
        super().__init__()
        self.stat_color_base_covariance = False
        self.stat_color_bandpass_covariance = False
        self.stat_color_low_covariance = True       # Raw (or uncentered) covariance of corresponding images from each color channel
        self.stat_color_edge_covariance = True
        self.stat_color_phase_covariance = True
        # Some named groups of statistics for convenience
        self.named_stat_groups = { 
                # categories based on statistics type
                 'crosscolor_stats':('color_low_covariance','color_edge_covariance','color_phase_covariance','color_base_covariance','color_bandpass_covariance'), 
                 }
                    
    # Compute each of the enabled statistics, average it over the pooling regions, and return
    # the result as a list of tensor images (one per statistic)
    # Can also record category and descriptive name for each statistic
    def forward(self,spyr_list,poolfunc,stat_labels,statlabel_callback=None):
        stats = []                                # List of statistics images to return
        # Base image is the image from which the steerable pyramid was built
        baseimg = spyr_list[0].original_image()
        basesize = baseimg.size()
        temporal = spyr_list[0].tname
        # Utility function to process and add one statistic image to list
        # statimg is the statistics image and cat is its category
        # src1,src2 are used to optinoally create descriptive strings for each statistic
        # raising to exponent is used to equalize the way statistics respond to any overall value scaling factors
        def add_stat(statimg,catname,chnames,level,*,ori=None,note=None):
            #if len(stats)==5: plot_image(statimg,title=name)
            weight = self.category_weights[catname]
            if self.per_level_weight: weight *= self._level_weight(level)
            stat = poolfunc.pool_stats(statimg,basesize)
            stats.append(weight*stat)
            if stat_labels is not None: 
                stat_labels.append(StatLabel(catname, level, chnames, temporal, note, ori))
                if statlabel_callback is not None:
                    statlabel_callback(stat,stat_labels[-1],statimg)
        def _ori_to_list(x): # returns list of tensors for each slice in dimension one, but keeps the same number of dimensions (unlike torch.unbind)
            return [x.narrow(1,ori,1) for ori in range(x.size(1))] if x is not None else None
        # Iterate over all pairs of channels
        for (a,b) in range_unique_pairs(len(spyr_list)):
            A = spyr_list[a]
            B = spyr_list[b]
            p = SPyramidParams.intersection(A.params(),B.params()) # use only levels present in both pyramids
            chnames = (A.cname,B.cname)
            if self.stat_color_base_covariance:
                add_stat(A.original_image()*B.original_image(),'covariance',chnames,level=None)
            if self.stat_color_bandpass_covariance:
                for i in p.bandpass_range():
                    add_stat(A.band_pass_image(i)*B.band_pass_image(i),'bandpass_variance',chnames,level=i)
            if self.stat_color_low_covariance:
                for i in p.lowpass_range():
                    add_stat(A.low_pass_image(i)*B.low_pass_image(i),'covariance',chnames,level=i)
            if self.stat_color_edge_covariance:
                for i in p.edge_range():
                    magAlist = _ori_to_list(A.edge_magnitude_images(i))
                    magBlist = _ori_to_list(B.edge_magnitude_images(i))
                    for ori,(magA,magB) in enumerate(zip(magAlist,magBlist)):
                        add_stat(magA*magB,'edge_covariance',chnames,level=i,ori=ori)
            if self.stat_color_phase_covariance:
                for i in p.edge_range():
                    erealAlist = _ori_to_list(A.edge_real_images(i))
                    erealBlist = _ori_to_list(B.edge_real_images(i))
                    for ori,(erA,erB) in enumerate(zip(erealAlist,erealBlist)):
                        add_stat(erA*erB,'phase_covariance',chnames,level=i,ori=ori,note='er')
        return stats     # Return list of statistic tensors (assume loss function can process a list)

""#END-CLASS------------------------------------



#print(f'all pairs: {list(range_all_ordered_pairs(4))}')
#print(f'unique pairs: {list(range_unique_pairs(4))}')
#print(f'distinct ordered pairs: {list(range_distinct_ordered_pairs(4))}')