#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:35:17 2019

The MetamerImageSolver implements the gradient descent solver used to generate
images with matching pooled statistics.  A MetamerImage holds the image to
be optimized (holds some extra information and allows some different image 
representations), a StatisticsEvaluator
is used to produce the pooled statistics from an image.  The solver will then
used gradients to modify the metamer image so that its pooled statistics more
closely match those of the target image.

The solver defaults to using the GPU to accelerate the statistic and gradient
computations.  A simple example of its use is given in the _test_solver() method
that is invoked if you run this file.  However the MetamerConfig class is
provides a simpler interface for configuring, initializing, and using 
this solver and is recommended for most uses. 

@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

# Takes a statistical model and a target image and then uses iterative learning
# (eg gradient descent) to generate a metamer image that matches its statistics

import torch
import spyramid as sp
import metamerstatistics as mstat
import metamerstateval as meval
import poolingregions as pool
import imageblends as blend
import metamerstatgroups
import os
import math
import time
import tempfile
import contextlib
import matplotlib.pyplot as plt
from image_utils import plot_image, save_image, load_image_gray, load_image_rgb, plot_images
try:
    import temporalstatistics 
    make_temporal_stat_evaluator = temporalstatistics.make_temporal_stat_evaluator
#    print("temporal experimental temporal extension was loaded")
except ImportError:
    make_temporal_stat_evaluator = None # Otherwise we will use the fallback code for still images

# Generalized MSE loss function that allows its input to be either lists of tensors or single tensors 
# Convenient if some of the tensors have different sizes and thus are not easily combined into a single tensor
def MSEListLoss(value,reference,scalefactor):
    # If not a list, assume its a tensor and just use standard mse_loss function
    if type(value) is not list:
        return torch.nn.functional.mse_loss(value,reference,reduction='sum')
    # Otherwise assume they are lists of tensors and compute MSE losses for each, and then sum
    losslist = [MSEListLoss(val,ref,scalefactor) for val,ref in zip(value,reference)]
    #print(f'losses: {losslist}')
    return scalefactor*sum(losslist)

# Similar to MSEListLoss above, except that it does not sum across elements or pixels but instead
# returns a tensor of the same size as the ones in the input
def SquaredDifferenceImage(value,reference,scalefactor,*,outputTensor=None):
    with torch.no_grad():
        if (type(value)) is list:
            for v,r in zip(value,reference):
                outputTensor = SquaredDifferenceImage(v,r,scalefactor,outputTensor=outputTensor)
        elif outputTensor is None:
            outputTensor = (value-reference)**2
        else:
            outputTensor.add_((value-reference)**2)
    return outputTensor


# Base class for a learned metamer.  By default we learn an image just like the target,
# but subclass can modify this to that the metamer image is generated/constrained in more
# complex ways (eg, upsampling a low-res image, or learned filter over existing image)
class MetamerImage(torch.nn.Module):
    
    def __init__(self,seed_image):
        super().__init__()
#        seed_image = seed_image.detach().clone()        # Make a copy so we don't overwrite the original
        if seed_image.dim()==3: seed_image=seed_image.unsqueeze(0)
        self.learned = torch.nn.Parameter(seed_image)   # Making it a parameter marks it as learnable
        self.learned.requires_grad_()                   # Metamer image is what we are trying to learn so we need its gradients
        self.frozen_mask = None                         # Optional mask indicating parts which are not subject to training
        # Some related data that can optionally be computed
        self.loss_value = math.nan                      # Loss value (or NaN if not yet computed)
        self.pooling_loss_image = None                  # Tensor with per-region pooling summed loss/error (or none if not computed)
        self.blame_image = None                         # Image the approximate local loss/error per pixel (or none if not computed)
        self.statgroup_loss_images = None               # Dictionary mapping StatGroups to group-keys to their loss images
        
    # Copy the specified pixels from the target image and force their gradients to be zero so optimizer will not modify them
    def copy_and_freeze_pixels_(self,target_image,copy_mask):
        with torch.no_grad():
            self.learned.copy_(torch.where(copy_mask,target_image,self.learned))
        #plot_image(self.learned)
        self.frozen_mask = torch.nn.Parameter(copy_mask,requires_grad=False)
        
    # Mark some pixels in metamer image as unchangeable (frozen) so optimizer cannot modify them
    def set_frozen_mask(self,freeze_mask):
        self.frozen_mask = torch.nn.Parameter(freeze_mask,requires_grad=False)       
        
    # Return a representation of the current state of this object
    def _get_current_state_copy(self):
        return self.clone_image()
    
    def _set_current_state(self,prior_state):
        with torch.no_grad():
            self.learned.copy_(prior_state)
    
    # Blend the current state with a prior state (from get_current_state_copy)
    def _blend_with_prior_state(self,prior_state,fraction):
        with torch.no_grad():
            self.learned.lerp_(prior_state, fraction)
        
    # Get a copy of the current metamer estimate suitable for visualization or saving (has no gradients and is on cpu)
    def get_image(self):
        image = self()
        if image.is_cuda:
            return image.detach().cpu()
        else:
            return image.detach().clone()  # Make a copy that does not affect gradients
        
    # Get a copy of current metamer estimate suitable for input to later iterations (no gradients but may not on gpu or cpu)
    def clone_image(self):
        return self().detach().clone();   # Make a copy on same device that does not affect gradients
        
    # Get a copy of the current gradient image suitable for visualization or saving 
    def get_gradient_image(self):
        image = self.learned.grad
        if image.is_cuda:
            return image.detach().cpu()
        else:
            return image.detach().clone()  # Make a copy that does not affect gradients

    # Get a copy of the current error image suitable for visualization or saving (size depends on number of pooling resgions)
    def get_pooling_loss_image(self,sqrt=False):
        image = self.pooling_loss_image
        if image is None: return None
        image = image.detach()         # Make sure image cannot affect gradients
        if sqrt: image = image.sqrt()  # Optionally take sqrt 
        if image.is_cuda:
            image = image.cpu()        # Copy to cpu before returning
        elif not sqrt:
            image = image.clone()      # Make a copy if not already done by prior steps
        return image
        
    # Get a copy of the current blame image (same size as metamer image or None if not computed)
    def get_blame_image(self):
        image = self.blame_image
        if image is None: return None
        if image.is_cuda:
            return image.detach().cpu()
        else:
            return image.detach().clone()  # Make a copy that does not affect gradients
                    
    def get_statgroups(self):
        if self.statgroup_loss_images is None: return ()  #empty tuple
        return self.statgroup_loss_images.keys()
    
    def get_statgroup_images(self,statgroup):
        return self.statgroup_loss_images[statgroup]
    
    def clear_auxiliary_data(self):
        self.loss_value = math.nan                      
        self.pooling_loss_image = None                  
        self.blame_image = None   
        self.statgroup_loss_images = None                   

    def set_loss_value(self,value):
        self.loss_value = value
    
    def set_pooling_loss_image(self,image):
        self.pooling_loss_image = image
    
    def set_blame_image(self,image):
        self.blame_image = image
        
    def set_statgroup_loss_images(self,statgroupdict):
        self.statgroup_loss_images = statgroupdict
        
    # Clamp current learned image to be no less than min and no more than max
    def clamp_range_(self,lower_limit,upper_limit):
        if (lower_limit is not None) or (upper_limit is not None): 
            with torch.no_grad():
                # Check if limits are lists (per-channel values) or constants (same across all channels)
                if (type(lower_limit) is list) or (type(upper_limit) is list):
                    for i in range(self.learned.size(-3)):
                        self.learned.narrow(-3,i,1).clamp_(min=lower_limit[i],max=upper_limit[i])                    
                else:
                    self.learned.clamp_(min=lower_limit,max=upper_limit)

    # Filter gradients for pixels with values at the lower or upper limits if gradient descent would lead to 
    # violating those limits and for pixels which have been marked as non-modifiable (ie non-trainable or constant)
    def clamp_range_gradients_(self,lower_limit,upper_limit):
        # If the image is at the min limit, then clamp its gradient to be non-positive (<=0) 
        # Idea is that the minimization is allowed to increase the element (if that will reduce the loss)
        # but not decrease it (since its value would just end up being clamped to min anyway) 
        # so we set gradient to zero for that case, so optimizer will not become confused by the clamping effect
        def clamp_at_min(img,limit):
            if limit is None: return
            with torch.no_grad():
                atmin = img.le(limit)              # Mask for tensor elements that are <= min
                mingrad = atmin.float()*img.grad   # Gradients of elements at min limit (and zero for other elements)
                badmingrad = mingrad.clamp(min=0)
                img.grad.sub_(badmingrad)          # Remove disallowed gradients from the gradient vector (in place)
        # Enforce similar limit for pixels >= max limit
        # Set their gradient to zero if it would otherwise imply that loss could 
        # be reduced by increasing their value beyond the max limit
        def clamp_at_max(img,limit):
            if limit is None: return
            with torch.no_grad():
                atmax = img.ge(limit)
                maxgrad = atmax.float()*img.grad
                badmaxgrad = maxgrad.clamp(max=0)
                img.grad.sub_(badmaxgrad)
        # Filter gradients for pixels <= lower_limit
        if type(lower_limit) is list:
            for i in range(self.learned.size(-3)):  # Apply limits to each color channel separately
                clamp_at_min(self.learned.narrow(-3,i,1),lower_limit[i])
        else:
            clamp_at_min(self.learned,lower_limit)  # Apply same limit to all
        # Filter gradients for pixels >= upper_limit
        if type(upper_limit) is list:
            for i in range(self.learned.size(-3)):  
                clamp_at_max(self.learned.narrow(-3,i,1),upper_limit[i])
        else:
            clamp_at_max(self.learned,upper_limit)
        # If some pixels are marked as frozen (not trainable) then set their gradients to zero
        if self.frozen_mask is not None:
            with torch.no_grad():
                self.learned.grad.masked_fill_(self.frozen_mask,0)

    # Returns the current estimate metamer image for use in gradient descent learning loop (may be on GPU or other device)
    def forward(self):
        # Metamer is just learned data "as is" here, but subclasses can override this
        return self.learned

""#END-CLASS------------------------------------    

# Experimental version where metamer is lower resolution than target image    
class LowResMetamerImage(MetamerImage):
    
    def __init__(self,seed_image,scale_factor=2):
        super().__init__(torch.nn.functional.avg_pool2d(seed_image,scale_factor,stride=scale_factor))
        self.scale_factor = scale_factor
        
    def copy_and_freeze_pixels_(self,target_image,copy_mask):
        raise NotImplementedError("cannot copy pixels to downsampled version")

    def get_lowres_image(self):
        image = self.learned
        if image.is_cuda:
            return image.detach().cpu()
        else:
            return image.detach().clone()  # Make a copy that does not affect gradients
        
    # Returns the current estimate metamer image for use in gradient descent learning loop
    def forward(self):
        return torch.nn.functional.interpolate(self.learned,scale_factor=self.scale_factor,mode='nearest')
        # Metamer is just learned data "as is" here, but subclasses can override this
        return self.learned
    
""#END-CLASS------------------------------------    

# Prints out some simple GPU memory usage statistics
def ms_print_gpu_mem():
    #print(torch.cuda.memory_summary())   #does not work for me, may be from a later release of pytorch?
    print(f'GPU alloc {torch.cuda.memory_allocated()/(2**30):.2f} GB  max {torch.cuda.max_memory_allocated()/(2**30):.2f} GB  reserved {torch.cuda.memory_reserved()/(2**30):.2f} GB')
#    torch.cuda.reset_max_memory_allocated()  #Reset max allocated so max is between each call to print
    torch.cuda.reset_peak_memory_stats()  #Reset max allocated so max is between each call to print

# Solve for a metamer image which matches the statistics of a given target image
# Here we refer to two images with matching statistics as metamers
# The metamer image is trained/learned using gradient descent until it has the desired statistics
# The statistics model can be global or regional (computed over pooling regions)
# Note: I'm not sure this is really a Module but that using that superclass provides some convenience methods
class MetamerImageSolver(torch.nn.Module):
    
    def __init__(self,statistics_evaluator):
        super().__init__()
        # Note: pytorch uses attributes (especially Parameters and Modules) to keep track of which data
        # needs to be moved the the GPU for gpu computations, its important to have these as instance attributes
        # Method for evaluating the statistics on a model
        self.stat_eval = statistics_evaluator
        # Target image whose statistics we want to match (only valid within solve() method)
        self.target_image = None
        # Current estimate of metamer image (only valid within the solve method)
        self.metamer = None
        # Optional preceding frames for target and metamer in case we are using temporal statistics *as tensors)
        # Optional sequences of prior frames extending backward in time
        self.target_prior_frames = None  
        self.metamer_prior_frames = None
#        self.target_prev_image = None
#        self.metamer_prev_image = None
        # Image values are constrained to be >= lower_limit (or use None for no limit)
        self.lower_limit = 0      
        # Image values are constrained to be <= upper_limit (or use None for no limit)
        self.upper_limit = None   
        # Use the gpu for computations if it is available
        self.use_gpu_if_available = True
        # We scale up the loss to avoid problems with low thresholds in the optimizer
        self.loss_scalefactor = 1e3
        # Some optional outputs that can be returned with(in) the metamer image
        self.return_gradient_image = False
        self.return_pooling_loss_image = False    # This is the per pooling region loss as a (reduced) image
        self.return_blame_image = False           # Image (same size as metamer) approximately projecting errors back to pixel locations
        self.return_loss_image_groups = []
        # Various optional debugging/progress printouts 
        self.print_num_statistics = True
        self.print_elapsed_time = True
        self.print_convergence_graph = False
        self.convergence_graph_groups = []    # Break out error by subgroups in this list such as 'level', 'category'
        self.print_loss = True
        self.print_loss_groups = []
        self.print_top_losses = False
        self.print_image = False
        self.print_image_comparison = False
        self.print_gradient_image = False
        self.print_pooling_loss_image = False
        self.print_category_loss_images = False
        self.print_blame_image = False
        self.print_gpu_memory = False
        self.save_image = False
        self.save_convergence_movie = False
        self.save_convergence_graph = False
        self.save_pooling_loss_image = False
        # These options applied at each step of the iteration
        self.step_print_image = False
        self.step_print_gradient_image = False
        self.step_print_pooling_loss_image = False
        self.step_print_blame_image = False
        self.step_print_loss = True
        self.step_print_loss_groups = []       # Print out loss by subgroups in this list such as 'level', 'category'
        self.step_print_top_losses = False
        self.step_print_gpu_memory = False
        self.step_save_image = False
        self.output_directory = ''
        self.statlabel_callback = None
        
    # A generic method to set an attribute but only if it already exists
    def set_mode(self,attr,value):
        if hasattr(self,attr):
            setattr(self,attr,value)
        else:
            raise NameError(f"Attribute: {attr}  does not exist or is misspelled")
            
    # A generic method for enabling/disabling specific statistics and name groups of statistics
    def set_stat_mode(self,stat,value):
        if type(value) is not bool:
            raise ValueError(f'Expected value of type bool but got: {value}')
        self.stat_eval.set_mode(stat,value)
        
    def get_statistics_evaluator(self):
        return self.stat_eval
            
    def constrain_image_min(self,min_value):
        self.lower_limit = min_value
        
    def constrain_image_max(self,max_value):
        self.upper_limit = max_value
        
    def constrain_image_range(self,target_image):
        num_channels = target_image.size(-3)  # Number of color channels in image
        if not (1 <= num_channels <= 3): raise ValueError(f'Image in invalid number of channels: {num_channels}')
        self.lower_limit = [target_image.narrow(-3,i,1).min().item() for i in range(num_channels)]                           
        self.upper_limit = [target_image.narrow(-3,i,1).max().item() for i in range(num_channels)]  
        
    # Prints the error for each statistic along with a label identifying the statistics
    def print_statistic_losses(self,stats,target_stats):
        with torch.no_grad():
            for i,(value,reference) in enumerate(zip(stats,target_stats)):
                err = self.loss_scalefactor*torch.nn.functional.mse_loss(value,reference,reduction='sum')
                print(f'{float(err):.3g} loss for {self.stat_eval.get_description(i)}')
            
    # Determine which statistics have the highest errors and print their description
    def _print_top_losses(self,stats,target_stats,count):
        if count is True: count = 1
        errlist = []
        with torch.no_grad():
            # Create list of pairs (statistic_index, statistic_loss) for all statistics
            errlist = [ (i, self.loss_scalefactor*( ((s-t)**2).sum().item() )) 
                       for i,(s,t) in enumerate(zip(stats,target_stats))]
            errlist.sort(reverse=True,key=lambda x: x[1])
        for i in range(count):
            print(f'{errlist[i][1]:.3g} loss for {self.stat_eval.get_description(errlist[i][0])}')
        
    # Computes errors within various subgroups such as pyramid_level, weight_category, or channel            
    def compute_loss_groups(self,stats,target_stats,statgroups):
        with torch.no_grad():
            result = metamerstatgroups.initialize_statgroup_dictionaries(statgroups)
            for i,(s,t) in enumerate(zip(stats,target_stats)):
                #err2 = ((s-t)**2).sum().item()
                err2 = torch.nn.functional.mse_loss(s,t,reduction='sum')
                label = self.stat_eval.get_label(i)
                for statgroup,lossdict in result.items():
                    key = statgroup.label_to_key(label)
                    lossdict[key] = lossdict.get(key,0) + err2   # Add loss to subgroup indicated by key (creating a new entry if needed)
        return result # return result dictionary which maps from statgroup to a mapping from key to summed error associated with that key
                    
    def print_loss_by_groups(self,stats,target_stats,statgroups):
        res = self.compute_loss_groups(stats,target_stats,statgroups)
        for sg in res:
            for key,val in res[sg].items():
                print(f'{sg.key_to_str(key)}: {val:.3g}')
                        
    def compute_loss_image_groups(self,stats,target_stats,statgroups):
        with torch.no_grad():
            result = metamerstatgroups.initialize_statgroup_dictionaries(statgroups)
            for i,(s,t) in enumerate(zip(stats,target_stats)):
                err2 = self.loss_scalefactor*(s-t)**2
                label = self.stat_eval.get_label(i)
                for statgroup,lossdict in result.items():
                    key = statgroup.label_to_key(label)
                    if key in lossdict:
                        lossdict[key].add_(err2)
                    else:
                        lossdict[key] = err2.clone()
        return result # return result dictionary which maps from statgroup to a mapping from key to summed loss image associated with that key
                
    # Print a graph fof the loss vs iteration (ie convergence)
    def plot_loss_convergence(self,losslist,show=True,savefile=None):
        plt.figure(figsize=[6,5])
        plt.title('Loss vs iteration')
        plt.semilogy(range(len(losslist)), losslist)
        plt.tight_layout()
        if savefile:
            plt.savefig(savefile, bbox_inches='tight')     # can be saved to .png or pdf file
            print(f'Convergence graph written to {savefile}')
        if show:
            plt.show()
        else:
            plt.close()
                    
    # Set output directory where any output, progress, or debugging files will be written
    def set_output_directory(self,path):
        path = os.path.expanduser(path)  #expand ~ or ~user to the user's home directory
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.isdir(path):
            raise ValueError(f'not a directory: {path}')
        self.output_directory = path
        
    # Get the output diretory for this solver
    def get_output_directory(self):
        return self.output_directory
    
    def forward(self):
        raise NotImplementedError("use solve method instead")
#        return self.stat_model(self.metamer)     # Compute and return statistics image for metamer

    # Solve for metamer of target_image using an iterative optimizer
    # This is the main metamersolver method and will automatically moved data to/from GPU if available
    def solve_for_metamer(self,target_image,max_iterations=10,seed_image=None,optimizer='LBFGS',lossfunc=None,copy_target_mask=None,
                          target_prior_frames=[],metamer_prior_frames=[]):
        # setup names and paths for output files
        outfilename = self.save_image                             # setup a name for the output image
        if type(outfilename)==bool: outfilename = 'result.png'
        if outfilename is None:
            outbasename = outbasepath = outfilepath = None
        else:
            outbasename = os.path.splitext(outfilename)[0]            # output name without the extension
            outbasepath = os.path.join(self.output_directory,outbasename)
            outfilepath = os.path.join(self.output_directory,outfilename)
        # Start a timer se we can print the computation time
        timer = time.perf_counter()
        # Setup target image and initial metamer image estimate
        if target_image.dim()==3: target_image = target_image.unsqueeze(0)  # Pytorch prefers four dimensions (batchXchannelXheightXwidth)
        if seed_image is None: 
            self.metamer = MetamerImage(torch.rand_like(target_image)) # Start with random noise image if none specified
        elif torch.is_tensor(seed_image):
            self.metamer = MetamerImage(seed_image.detach().clone())    # If image was provided use it as intial metamer estimate
        else:
            self.metamer = seed_image;                 # assume seed image was an already configured metamer image
        if copy_target_mask is not None and copy_target_mask is not False:
            # Copy the specified pixels from the target image and force their gradients to be zero so optimizer will not modify them
            self.metamer.copy_and_freeze_pixels_(target_image,copy_target_mask)
        self.metamer.clamp_range_(self.lower_limit,self.upper_limit)      # Clamp any out-of-range pixels in the input
        # Register images if they were supplied (as not needing gradients and copied to GPU if needed)
        self.target_image = torch.nn.Parameter(target_image,requires_grad=False)
        if target_prior_frames or metamer_prior_frames:
            # need to add prior images to parameter list so that they will get copied to GPU if needed
            if not target_prior_frames: raise ValueError('Missing previous frames: {target_prior_frames}')
            if not metamer_prior_frames: raise ValueError('Missing previous metamers: {metamers_prior_frames}')
            if len(target_prior_frames)!=len(metamer_prior_frames): raise ValueError("Did not provide equal length frame histories for target and metamer")
            self.target_prior_frames = torch.nn.ParameterList([torch.nn.Parameter(img,requires_grad=False) for img in target_prior_frames])
            self.metamer_prior_frames = torch.nn.ParameterList([torch.nn.Parameter(img,requires_grad=False) for img in metamer_prior_frames])
#            self.target_prev_image = torch.nn.Parameter(target_prev_image,requires_grad=False)
#            self.metamer_prev_image = torch.nn.Parameter(metamer_prev_image,requires_grad=False)
        else:
            self.target_prior_frames = []  # set them to empty list just to make later code simpler
            self.metamer_prior_frames = []
        # Move our attribute tensors and module attributes to the GPU (module keeps track of its member attributes).
        # This should happen before constructing the optimizer in case the optimizer constructs internal tensors
        if self.use_gpu_if_available and torch.cuda.is_available():
            print("Using GPU for solver computations")
            self.cuda()   
                
        # Setup the loss and optimzation functions (if not already specified)
        if lossfunc is None:
            #lossfunc = torch.nn.MSELoss(reduction='sum')
            lossfunc = MSEListLoss
        elif self.print_regional_loss or self.save_regional_loss or self.step_print_regional_loss:
            print("Warning: regional loss plots may not match specified custom loss function")
        trainables = filter(lambda p: p.requires_grad, self.parameters())  # Trainable parameters (should just be metamer)
#        if optimizer == 'AdaptiveStepRootFinder':
#            optimizer = rootfinderopt.AdaptiveStepRootFinder(trainables,momentum=0.75)
        if optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(trainables, max_iter=1, history_size=30, tolerance_grad=1e-9)  
#            optimizer = torch.optim.LBFGS(trainables, max_iter=1, history_size=30, tolerance_grad=1e-12, tolerance_change=1e-12)  
#            optimizer = torch.optim.LBFGS(trainables, max_iter=1, history_size=30, tolerance_grad=1e-9, line_search_fn='strong_wolfe',max_eval=8)  
#            optimizer = torch.optim.LBFGS(trainables, lr=1, max_iter=1, history_size=50, tolerance_grad=0, tolerance_change=0)  
#            optimizer = torch.optim.LBFGS(trainables, lr=1, max_iter=1, history_size=50, tolerance_grad=1e-12, tolerance_change=1e-12)  
        else:
            raise RuntimeError(f'Unrecognized optimizer type: {optimizer}')
        
        # Compute the statistics for the target image, this is what we will try to match
        #  for the input we construct list of frames starting with the current and going backward in time
        target_stats = self.stat_eval([self.target_image, *self.target_prior_frames],create_labels=True,statlabel_callback=self.statlabel_callback)
        
        if self.print_num_statistics: print(f"Total number of statistics: {len(target_stats)}")
        losslist = []      # List of loss values at each step (so we can plot or analyze them later)
        # Related quantities that we may need to compute (if they were requested)
        keepBlameImage = self.step_print_blame_image
        keepPoolingImage = self.step_print_pooling_loss_image
        loss_image_groups = []
        print_loss_groups = self.step_print_loss_groups
        print_top_losses = self.step_print_top_losses
        saved_met_state = None
        
        # Define loss evaluation as a function so optimizer can call it and get loss (depends on optim used if this is required)
        def closure(compute_gradients=True,max_retries=2):
            nonlocal saved_met_state
            self.metamer.clear_auxiliary_data()    # Clear any old auxiliary data from metamer
            optimizer.zero_grad()                  # Clear any gradients from prior computations
            stats = self.stat_eval([self.metamer(), *self.metamer_prior_frames]) 
                                                   # Evaluate the model on the current estimate and return its statistics
            loss = lossfunc(stats, target_stats,   # Loss measures difference between statistics
                            self.loss_scalefactor) #  scaled up to avoid triggering epsilon thresholds in the optimizer
            # Check if loss increased significantly and retry with smaller step in that case
            if losslist and (not (1.001*losslist[-1] > loss)):
                if (max_retries>0):
                    if self.step_print_loss: print(f'Loss increased to {loss:.3g}, retrying again with smaller step')
                    self.metamer._blend_with_prior_state(saved_met_state, 0.75) #Blend with old state to simulate smaller step size
                    optimizer.state[optimizer._params[0]]['t'] *= 0.25  #Tell optimizer about smaller step size
                    del stats  # allow memory to be garbage collected
                    del loss
                    return closure(compute_gradients,max_retries-1)
                else:
                    if self.step_print_loss: print(f'Failed to reduce loss, restarting with cleared history')
                    self.metamer._set_current_state(saved_met_state)
                    optimizer.state[optimizer._params[0]]['n_iter'] *= 0
                    del stats  # allow memory to be garbage collected
                    del loss
                    return closure(compute_gradients)
            else:
                saved_met_state = self.metamer._get_current_state_copy()
            losslist.append(float(loss))           # Add loss value to list in case we want it later
            if compute_gradients:
                loss.backward()                    # Compute image gradients with respect to loss
                                                   # Adjust gradients for any clamped or out-of-range pixels
                self.metamer.clamp_range_gradients_(self.lower_limit,self.upper_limit)          
#                if torch.isnan(self.metamer.learned.grad).any(): print(f'Nan in gradient image')
            # compute any requested optional quantities that depend on the stats 
            if keepPoolingImage or keepBlameImage: self.metamer.set_pooling_loss_image(SquaredDifferenceImage(stats,target_stats,self.loss_scalefactor))
            if keepBlameImage: self.metamer.set_blame_image(self.stat_eval.blame_stats(self.metamer.pooling_loss_image))
            if loss_image_groups: self.metamer.set_statgroup_loss_images(self.compute_loss_image_groups(stats,target_stats,loss_image_groups))
            if print_loss_groups: self.print_loss_by_groups(stats,target_stats,print_loss_groups)
            if print_top_losses: self._print_top_losses(stats,target_stats,print_top_losses)
            return loss
        
        # Create a context manager so temporary directories or files are deleted at the end of this scope
        with contextlib.ExitStack() as cmscope:
            if self.step_save_image:  # create a directory to save step images or output to
                iterdir = f'{outbasepath}steps{int(time.time())}'
                os.mkdir(iterdir)
            elif self.save_convergence_movie and (outbasename is not None):
                tmpdir = tempfile.TemporaryDirectory(prefix=outbasename+'temp',dir=self.output_directory)
                iterdir = tmpdir.name
                cmscope.enter_context(tmpdir)  #ensure temporary directory will be deleted at end of scope
                
            # Learning loop to train the metamer
            for inum in range(max_iterations):
                if self.step_print_image: plot_images(self.metamer.get_image(),center_zero=False,title=f'Step {len(losslist)} Image')
                optimizer.step(closure)     # Invoke optimizer to perform one optimization iteration
                self.metamer.clamp_range_(self.lower_limit,self.upper_limit)  # Clamp any out-of-range pixels that optimizer might have created
                if self.step_print_gradient_image: plot_image(self.metamer.get_gradient_image(),title=f'Step {len(losslist)} Gradient Image')
                if self.step_print_pooling_loss_image: plot_image(self.metamer.get_pooling_loss_image(),title=f'Step {len(losslist)} Regional Loss')
                if self.step_print_blame_image: plot_image(self.metamer.get_blame_image(),title=f'Step {len(losslist)} Blamed Loss')
                if self.step_print_loss: print(f'Step {len(losslist)} loss: {losslist[-1]}')
                if self.step_print_gpu_memory: ms_print_gpu_mem()
                if (self.step_save_image or self.save_convergence_movie) and (outbasename is not None):
                    save_image(self.metamer.get_image(),os.path.join(iterdir,f'metamer_iter{inum:03d}.png'))
            ""#End-of-learning-loop--------------------------            
            # Compute the loss of the final result (and any needed related quantities)
            keepBlameImage = self.return_blame_image or self.print_blame_image
            keepPoolingImage = self.return_pooling_loss_image or self.print_pooling_loss_image or self.save_pooling_loss_image
            loss_image_groups = self.return_loss_image_groups
            keepGradients = self.return_gradient_image or self.print_gradient_image
            print_loss_groups = self.print_loss_groups
            print_top_losses = self.print_top_losses
            if max_iterations >= 0:
                closure(compute_gradients=keepGradients)    # Compute statistics for final result
            else:
                print("Not computing statistics for metamer because number of iterations was negative")
            
            timer = time.perf_counter() - timer  # stop timer (don't include result saving or movie generation)
            if self.print_elapsed_time: print(f"Elapsed solver time {timer} secs")
            
            result = self.metamer  # This is value we will return at the end of this function
            result.loss_value = losslist[-1] if len(losslist)>0 else None  # Store final loss 
            # print/save any desired outputs before returning the result
            if self.print_loss: print(f'Final loss: {result.loss_value}')
            if self.print_gpu_memory: ms_print_gpu_mem()
            if len(losslist)>0 and (self.print_convergence_graph or self.save_convergence_graph):                
                graphfile = self.save_convergence_graph
                if graphfile is True: 
                    if outbasepath is not None: 
                        graphfile = f'{outbasepath}_lossgraph.pdf'
                    else:
                        graphfile = False 
                self.plot_loss_convergence(losslist,show=self.print_convergence_graph,savefile=graphfile)
            if self.print_category_loss_images:
                for cat,img in result.category_loss_images.items():
                    plot_image(img,title=f'Loss for {cat}')
            if self.print_pooling_loss_image or self.save_pooling_loss_image:
                graphfile = self.save_pooling_loss_image
                if graphfile is True: graphfile = f'{outbasepath}_regionloss.pdf'
                plot_image(result.get_pooling_loss_image(),show=self.print_pooling_loss_image,savefile=graphfile,title='Pooling Regions Loss');
            if self.print_blame_image:
                plot_image(result.get_blame_image())
            if self.print_gradient_image: plot_image(result.get_gradient_image())
            if self.print_image: plot_image(result.get_image(),center_zero=False,title='Metamer Image')
            if self.print_image_comparison: plot_images(torch.cat((result.get_image(),target_image),-1),center_zero=False,title='Metamer & Target Image')   
            if self.save_image: 
                save_image(result.get_image(),outfilepath)
            if self.save_convergence_movie and (outbasename is not None): # use ffmpeg to compile the iterations into a movie
                moviename = f'{outbasepath}_converge.mp4'
                blend.compile_frames_to_mp4(os.path.join(iterdir,'metamer_iter%03d.png'),moviename)
        ""# Temporary folder is deleted here (end of with scope)         
        self.metamer_frame_seq = None       #Clear some temporary fields in this object before returning
        self.target_frame_seq = None               
        return result   
    
""#END-CLASS------------------------------------

# Initialize a metamer solver with a default set of configuration values
def make_solver(target_image,           # target image, size is used to precompute some filters and optionally to set pixel range constraints
                stat_pooling,           # statistics pooling, can be a size or an preconfigured pooling object or a PoolingParams object
                outfile='pmetamer.png', # Filename where solver will eventually write result (or can be False)  
                prefilter = None,       # Optional prefilter to be applied before pyramid construction
                max_value_constraint=1, # Constrain max pixel value (set to None for no max constraint)
                pyramid_builder=None,   # Optionally provide a specific steerable pyramid builder (for expert use only, overrides other pyramid settings)
                pyramid_params=None,    # Specifies which pyramid levels to build for each image (can vary by channel and if set takes precedence over other parameters below)
                temporal_mode=None,     # Mode for handling temporal information (default is only consider current frame)
                ):   
    # Convert any params that are given as strings or other non-canonical forms
    pyramid_params = sp.SPyramidParams.normalize(pyramid_params)
    if pyramid_params is None: pyramid_params = sp.SPyramidParams(4)  # Old default value, maybe should be removed?
    print(f'Creating solver: pyramid={str(pyramid_params)} pooling={str(stat_pooling)}')
    # Choose the statistic pooling region
    if isinstance(stat_pooling,(int,float)):
        # assume pooling is size (length/diameter) so use our default separable kernel shape with that size
        stat_pooling = pool.Trigezoid(stat_pooling)
    elif isinstance(stat_pooling,str):
        params = pool.PoolingParams.from_str(stat_pooling)
        stat_pooling = params.to_pooling()
    elif hasattr(stat_pooling,'to_pooling'):
        stat_pooling = stat_pooling.to_pooling()  # convert PoolingParam object to a pooling kernel
    elif hasattr(stat_pooling, 'pool_stats'):        
        pass # assume it is an already configure statistic pooling object so just use it
    else:
        raise ValueError(f'expected pooling size or object, but got {stat_pooling}')
    # If no pyramid builder was specified, then configure a default builder
    if pyramid_builder is None:
        # automatically determine a max_downsampling such that windows still land on integer coordinates
        min_pool_spacing = stat_pooling.min_stride_divisor()
        max_downsample = math.gcd(math.gcd(64,min_pool_spacing),math.gcd(target_image.size(-1),target_image.size(-2)))
        #print(f"max_downsample_factor: {max_downsample}")
        stat_pooling.configure_for_downsampling(max_downsample)
        # Create the steerable pyramid builder.  It may use the target image size to precompute its filters
        # Downsampling causes the coarser levels to be stored at lower resolution (save memory and computation)
        # We limit the max downsampling because pooling windows use integer shifts and thus cannot be less than the coarsest resolution used in the pyramid
        # Future: add code to automatically set max_downsample_factor based on the pooling
        pyramid_builder = sp.SPyramidFourierBuilder(target_image,pyramid_params,
                                                    downsample=True,max_downsample_factor=max_downsample)
    if make_temporal_stat_evaluator is not None:
        stat_evaluator = make_temporal_stat_evaluator(temporal_mode,pyramid_builder,stat_pooling,prefilter)
    else:
        # fallback code for still images if experimental temporal extension code was not found
        # Setup the intra-channel statistics (eg portilla&simoncelli, etc.)
        channelstats = mstat.GrayscaleStatistics(autocorrelation_window=7)
        # Setup cross channel stats (for color images)
        crosscolorstats = mstat.CrossColorStatistics()
        if (temporal_mode is None) or (temporal_mode=='still'):
            # Create a still image evaluator (just current frame)
            stilleval = meval.FrameChannelStatisticsEvaluator('',None,channelstats,pooling=stat_pooling,pyramid_builder=pyramid_builder,crosscolor_stats=crosscolorstats)
            tchannel_evals = [stilleval]
            crosst_evals = []
        else:
            raise ValueError(f'Unrecognized temporal mode: {temporal_mode}  (note: temporalstatistics extension code was not found)')
        if temporal_mode is not None:
            print(f'setting temporal mode to: {temporal_mode}')
        # Now combine these into a image statistics evaluator
        stat_evaluator = meval.StatisticsEvaluator(tchannel_evals,crosst_evals,prefilter=prefilter)
    # Create a solver and configure it with some reasonable defaults
    solver = MetamerImageSolver(stat_evaluator)
    # Pixels are required to be positive and we can add more constraints to their allowed range
    if max_value_constraint:
        solver.constrain_image_max(max_value_constraint)
    # Turn on or off various informational outputs in the solver
    solver.set_mode('print_num_statistics',True)
    solver.set_mode('save_image',outfile)
    solver.set_mode('use_gpu_if_available',True)
    solver.set_output_directory('~/pmetamer_output')
    # Return the configured solver.  To generate the metamer invoke:
    # metamer_image = solver.solve_for_metamer(target_image,max_iterations,seed_image)
    return solver

# A simple example of metamersolver usage that generates a quick unconverged metamer    
def _test_solver():
    color = True
    whole_image_pool = False
    # Load a target image to match
    image_file = '../sampleimages/cat256.png'
    if color:
        target_image = load_image_rgb(image_file)
    else:
        target_image = load_image_gray(image_file)
    if whole_image_pool:
        statpool = pool.WholeImagePooling()
    else:
        statpool = 64     # Use 64x64 pooling regions tiled over the image (with overlap)  
    
    solver = make_solver(target_image,statpool)  # Create the metamer solver
    solver.set_mode('step_print_image',True)     # Solver can be configured in various ways
    solver.set_mode('save_image',False)
    if not whole_image_pool: solver.set_mode('step_print_pooling_loss_image',True)
    torch.manual_seed(49832475)   # This makes generation of the seed image deterministic
    seed_image = None             # Seed image will be initialized with random noise
    num_iterations = 16            # Choose the number of iterations (should be at least hundreds for good results)
    # Now solve for the metamer
    metamer_image = solver.solve_for_metamer(target_image,num_iterations,seed_image)
    # and plot the output
    plot_image(torch.cat([metamer_image.get_image(),target_image],-1))
    solver.stat_eval.print_stat_states()
#    save_image(metamer_image.get_image(),'cat_normal.png')
#    for label in solver.stat_eval.get_all_labels():
#      print(label)
#    print(f'Modes: {solver.stat_eval.get_list_of_modes()}')
    
if __name__ == "__main__":   # execute main() only if run as a script
    _test_solver()
