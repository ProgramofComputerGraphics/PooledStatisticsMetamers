#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:27:48 2020

Some utlity classes for grouping statistics into groups.  Useful when for reporting
or analyzing groups of statistics.


@author: bw
"""
# This code is part of the PooledStatisticsMetamers project
# Released under an open-source MIT license, see LICENSE file for details

# A StatGroup provides methods for grouping statistics into groups based on shared properties in their labels
# Used for reporting loss broken out into different components
class StatGroup():
    def label_to_key(self,label):
        raise NotImplementedError('Subclasses must implement this method')
    def key_to_str(self,key):
        return key
""#END-CLASS------------------------------------

class StatGroupLevels(StatGroup):
    def label_to_key(self,label): return label.level
    def key_to_str(self,key): return f'Level {key}'
    
class StatGroupCategories(StatGroup):
    def label_to_key(self,label): return label.weight_category
    
class StatGroupChannels(StatGroup):
    def label_to_key(self,label): return label.channel
    def key_to_string(self,key): return f'Channel {key}'
    
class StatGroupTypes(StatGroup):
    def label_to_key(self,label):
        if 'edge' in label.weight_category: return 'edge'
        if 'phase' in label.weight_category: return 'phase'
        return 'value'
    
class StatGroupIndividual(StatGroup):
    def label_to_key(self,label): return str(label)
    
class StatGroupCombo(StatGroup):  #Allows combining StatGroups by concatenating their keys
    def __init__(self,grouplist):
        self.grouplist = grouplist
    def label_to_key(self,label): 
        key = ''
        for sg in self.grouplist:
            if key: key = key + ' '
            key = key + str(sg.label_to_key(label))
        return key

# Filter statistics by some categories and report matching ones as a group or individuals
class StatGroupFilter(StatGroup):
    def __init__(self,level=None,temporal=None,channel=None,group=False):
        self.level = level
        self.temporal = temporal
        self.channel = channel
        self.group = group
    def label_to_key(self,label): 
        other = 'other'
        if self.level is not None and self.level != label.level: return other
        if self.temporal is not None and self.temporal != label.temporal: return other
        if self.channel is not None and self.channel != label.channel: return other
        if self.group: return 'group'
        return str(label)
    def key_to_string(self,key): 
        if self.group:
            if self.channel is not None: key += ' '+self.channel
            if self.level is not None: key += ' L'+self.level
            if self.temporal is not None: key += ' time'
        return key        
    
""#END-CLASS------------------------------------

statgroup_aliases = {'levels':StatGroupLevels(), 'categories':StatGroupCategories(),
                  'channels':StatGroupChannels(), 'types':StatGroupTypes(),
                  'individuals':StatGroupIndividual()}

def get_statgroup_by_name(name):
    return statgroup_aliases[name]

# Resolve any aliases and create a dictionary mapping each statgroup to an empty dictionary
def initialize_statgroup_dictionaries(statgroups):
    if not isinstance(statgroups,(list,tuple)): statgroups = (statgroups,)  # Make sure statgroups is a tuple/list
    sgdict = {}
    for sg in statgroups:
        if isinstance(sg,(tuple,list)):  # Convert a list/tuple to a combo stat group (with alias lookup)
            sg = StatGroupCombo([statgroup_aliases.get(g,g) for g in sg])
        else:                 
            sg = statgroup_aliases.get(sg,sg)
        if not getattr(sg,'label_to_key',None): raise ValueError(f'Unrecognized StatGroup: {sg}')
        sgdict[sg] = {}  # initialize set of key mappings to empty
    return sgdict
