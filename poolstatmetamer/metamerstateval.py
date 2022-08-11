#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:51:02 2021

This file contains some helper classes for evaluating statistics on an image
or set of images from a movie sequence.  It will build the steerable pyramids
for each channel&image and invoke objects to compute the intra-channel and
cross-channel statistics.

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

import torch
import color_utils as color

# Simple temporal filter (weighted combination of current and prior frames)
class WeightedImageFilter(torch.nn.Module):

    def __init__(self,filter_weights):
        super().__init__()
        self.filter_weights = filter_weights
        
    def forward(self,images):
        retval = None
        for weight,img in zip(self.filter_weights,images):
            if weight == 0: continue  # don't bother if known to be zero anyway
            if retval is None:
                retval = weight*img
            else:
                retval = weight*img + retval
        return retval
    
    # Return the number of prior frames used by this filter (count does not include current frame)
    def get_history_size(self):
        return len(self.filter_weights) - 1
    
# Simple temporal filter (return fixed frame from the list of frames, 0 is current frame, 1 is previous frame, etc.)
class IndexedImageFilter(torch.nn.Module):

    def __init__(self,index):
        super().__init__()
        self.index = index
        
    def forward(self,images):
        if self.index >= len(images): return None
        return images[self.index]
    
    def get_history_size(self):
        return self.index
    

# This object computes statistics within a single frame.  This "frame" could be a still image, 
# a single frame from a sequence (ie the current frame), or a "synthetic" frame generated from
# some combination of the current and prior frames from a movie sequence
class FrameChannelStatisticsEvaluator(torch.nn.Module):

    def __init__(self,name,temporal_filter,channel_stats,pooling,pyramid_builder=None,crosscolor_stats=None,
                 pyramid_crossscale=True,color_transform=color.RGBToOpponentConeTransform()):
        super().__init__()
        if temporal_filter is None: 
            temporal_filter = IndexedImageFilter(0)
        elif isinstance(temporal_filter,int): 
            temporal_filter = IndexedImageFilter(temporal_filter)
        elif isinstance(temporal_filter,(list,tuple)): 
            temporal_filter = WeightedImageFilter(temporal_filter)
        self.name = name
        self.temporal_filter = temporal_filter
        self.colorspace = color_transform  # Transform to colorspace used for statistics (if its a color image)
        self.builder = pyramid_builder     # Module that builds a steerable pyramids from images
        self.builder_crossscale = pyramid_crossscale # Generate phase-doubled images used for cross-scale correlations?
        self.channel_stats = channel_stats # Module that computes intra-channel statistic images from steerable pyramids
        self.crosscolor_stats = crosscolor_stats # Module that computes inter-color-channel statistic images (if it is is a color image)
        self.poolfunc = pooling            # Statistic pooling function (typically blurs and downsamples stat images)
    
    # Compute the statistics for the given images and return them as a list of (pooled) images
    #   images is a list of frames starting with the current frame and going backward in time (if prior frames are available to the statistics)
    #   if stat_labels is a list then labels will be added to it for each statistic added
    def forward(self,images,*,stat_labels=None,statlabel_callback=None):
        spyr_list = []   # List of steerable pyramids for each channel
        stats_list = []       # List of all statistic images
        image = self.temporal_filter(images)  # apply temporal filter to create image
        if image is None:
            return (stats_list,spyr_list) # may be none if no images are within the temporal support of the filter
        # Transform to statistics color space (if it is a color image)
        if image.size(1) == 3:                # Transform to specified colorspace if it is a color image
            image = self.colorspace(image)
            channel_names = self.colorspace.channel_names()
        elif image.size(1) == 1:
            channel_names = ('',) # No need for channel names if there is only one
        else: 
            raise NotImplementedError(f'Only 1-grayscale or 3-rgb channel images supported.  Number channels was {image.size(1)}')
        def eval_stat(evaluator,spyrs):
            stats = evaluator(spyrs,poolfunc=self.poolfunc,stat_labels=stat_labels,statlabel_callback=statlabel_callback)   #evaluate the statistics
            stats_list.extend(stats)                                    #add stats to list of all stats
        # For each channel compute the intra-channel statistics
        for c in range(image.size(1)):
            spyr = self.builder.build_spyramid(image.narrow(1,c,1),colorname=channel_names[c],temporalname=self.name,make_crossscale=self.builder_crossscale)
            spyr_list.append(spyr)
            eval_stat(self.channel_stats,spyr)
        # Compute cross channel statistics (if image has more than one channel)
        if (self.crosscolor_stats is not None) and (len(spyr_list) > 1):
            eval_stat(self.crosscolor_stats,spyr_list)
        # Return the list of statistics images
        return (stats_list,spyr_list)
    
    # Returns all the statistic objects (subclasses of MetamerStatistics) used by this evaluator
    def stat_objects(self):
        yield self.channel_stats
        if self.crosscolor_stats is not None: yield self.crosscolor_stats
        
    # Return the maximum number of prior frames used when evaluated this temporal channel
    def max_prior_frames_used(self):
        return self.temporal_filter.get_history_size()

""#END-CLASS------------------------------------    

    

# Evaluate and returns the statistics for an image.  Can contain multiple frame-channels and cross-channel statistics
# Generally though there will only be a single frame channel unless using the experimental temporal extension on a movie sequence
class StatisticsEvaluator(torch.nn.Module):
    
    def __init__(self,temporal_evals,cross_evals=[],prefilter=None):
        super().__init__()
        self.prefilter = prefilter        # Can be None or a Module that will apply some prefiltering to the images before computing statistics
        self.temporal_evals = torch.nn.ModuleList(temporal_evals)            # List of temporal channel statistics evaluators
            
        self.statlabels = None            # List of StatLabels for each statistic (subclass of NamedTuple)
        self.cross_evals = torch.nn.ModuleList(cross_evals)   # List of cross-temporal-channel statistics evaluators
        
    # Generate and return the steerable pyramids for this image but don't generate any statistics
    def make_pyramids(self,images):
        raise NotImplementedError("not yet implemented")
        # Apply prefilter to images (if specified)
        if self.prefilter is not None:       # Apply prefilter if specified
            images = [self.prefilter(img) for img in images]
        
    # Compute the statistics for the given images and return them as a list of (pooled) images
    # Images is a list of frames starting with the current frame and going backward in time 
    def forward(self,images,*,create_labels=False,statlabel_callback=None):
        spyr_dict = {}      # Dictionary from temporal channel names to its steerable_pyramids 
        stats_list = []     # List of all statistic images
        stat_labels = None  # None means don't bother collecting the StatLabels
        if create_labels:
            stat_labels = []  #collect set of statistic labels on first run (will be appended to this list)
        if torch.is_tensor(images): images = (images,)  # make into a tuple/list if just a single image is given
        # Apply prefilter to images (if specified)
        if self.prefilter is not None:       # Apply prefilter if specified
            images = [self.prefilter(img) for img in images]
        # Evaluate each of the temporal channels and any statistics within that channel
        for teval in self.temporal_evals:
            (stats,spyrs) = teval(images,stat_labels=stat_labels,statlabel_callback=statlabel_callback)
            stats_list.extend(stats)
            spyr_dict[teval.name] = spyrs
        # Evaluate any cross temporal channel statistics
        for xeval in self.cross_evals:
            stats = xeval(spyr_dict,stat_labels=stat_labels,statlabel_callback=statlabel_callback)
            stats_list.extend(stats)
        # Return the list of statistics images
        if stat_labels is not None: self.statlabels = stat_labels # If generated, save label list for later access
        return stats_list
    
    def max_prior_frames_used(self):
        return max(teval.max_prior_frames_used() for teval in self.temporal_evals)

    # Returns all the statistic objects (subclasses of MetamerStatistics) used by this evaluator
    def stat_objects(self):
        for teval in self.temporal_evals:
            yield from teval.stat_objects()
        for xeval in self.cross_evals:
            yield from xeval.stat_objects()
    
    # Turn on/off a statistics mode
    def set_stat(self,attr,value):
        count = 0
        for statobj in self.stat_objects():
            if statobj.set_stat(attr,value,False): count += 1
        if count == 0:
            # Check and see if it was a named group of statistics instead
            for statobj in self.stat_objects():
                if statobj.set_stat_group(attr,value): return
            raise NameError(f"Statistic mode: {attr} does not exist or is misspelled")
        if count > 1:
            print(f'Warning statistic mode {attr} was present in more than one statistics object')
    
    # Turn on/off all the available statistics
    def set_all_stats(self,value):
        for statobj in self.stat_objects():
            statobj.set_all_stats(value)
            
    def get_list_of_stats(self):
        statlist = []
        for statobj in self.stat_objects():
            statlist += statobj.get_list_of_stats()
        return statlist
    
    def print_stat_states(self):
        print('{',end='')
        for s in self.stat_objects():
            for m in s.get_list_of_stats():
                val = s.get_stat(m)
                print(f'{m}:{val} ',end='')
        print('}')
            
    # Statistics will have their weight multiplied by value^level
    # Deprecated:  use set_mode('per_level_weight',value) instead, may be removed in future versions
    def set_per_level_weight_multiplier(self,value):
        for s in self.stat_objects():
            s.set_per_level_weight_multiplier(value)
            
    # General interface for changing other statistic-related settings (not for enabling/disabling individual statisitcs use set_stat() for that)
    def set_mode(self,name,value):
        for s in self.stat_objects():
            s.set_mode(name,value)
            
    # Returns StatLabel for the specified statistic  (which is a subclass of NamedTuple)
    def get_label(self,statindex):
        return self.statlabels[statindex]
    
    # Return a list of all StatLabels (one for each statistic in same order as in statistics list)
    def get_all_labels(self):
        return self.statlabels
    
    # Get a brief descriptive string for the specified statistic
    def get_description(self,statindex):
        return str(self.statlabels[statindex])
    
    def get_category(self,statindex):
        return self.statlabels[statindex].weight_category
                
    def blame_stats(self,stat_image,original_size):
        raise NotImplementedError("not yet implemented")
#        with torch.no_grad():
#            image = self.poolfunc.blame_stats(stat_image,original_size) 
#        return image

""#END-CLASS------------------------------------    
