#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 20:07:02 2020

@author: ghiggi
"""
#-----------------------------------------------------------------------------.
### Utils for parsing strings ####
# https://www.rdocumentation.org/packages/stringr/versions/1.4.0 
import re 
import ast
import numpy as np 

def rep(x, times=1, each=False, length=0):
    """
    Implementation of functionality of rep() and rep_len() from R.
    Attributes:
        x: numpy array, which will be flattened
        times: int, number of times x should be repeated
        each: logical; should each element be repeated 'times' before the next
        length: int, length desired; if >0, overrides 'times' argument
    """
    flag = "numpy"
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        flag = "list"
    if length > 0:
        times = np.int(np.ceil(length / x.size))
    x = np.repeat(x, times)
    if(not each):
        x = x.reshape(-1, times).T.ravel() 
    if length > 0:
        x = x[0:length]
    if flag == "list":
        x = x.tolist()
    return(x)
                         
def rep_len(x, length_out):
    return np.resize(x, length_out)                              

def subset_list_by_index(x, l_index):
    return list(np.array(x)[np.array(l_index)])

def subset_list_by_boolean(x, l_boolean):
    # [i for (i, logic) in zip(x, l_boolean) if logic]
    # np.array(x)[np.array(l_boolean)].tolist()
    import itertools  
    return list(itertools.compress(x, l_boolean))
    
def str_simplify(x, simplify=True, empty_pattern=''): 
    """
        If x is a list with just 1 string element, return a string 
        If x is an empty list, return a '' string 
        If x is a string, return x
    """
    if (simplify is True):
        if isinstance(x, list):
            if len(x) == 0: 
                return empty_pattern
            elif len(x) == 1:
                return x[0]
            else:
                return x
        else:
            return x 
    else: 
        return x 

def isfloat(s): 
    """Return a boolean indicating if the string can be converted to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False  
    
    
def isinteger(s): 
    """Return a boolean indicating if the string can be converted to float."""
    if (isfloat(s)):
        return float(s).is_integer()
    else:  
        return False     

def str_isfloat(l_string):
    """Return a boolean indicating if the string can be converted to float."""
    if isinstance(l_string, list):
        return [isfloat(string) for string in l_string]  
    else: 
        return isfloat(l_string)

def str_isinteger(l_string):
    """Return a boolean indicating if the string can be converted to float."""
    if isinstance(l_string, list):
        return [isinteger(string) for string in l_string]  
    else: 
        return isinteger(l_string)
    
def islist(s): 
    """
    Return a boolean indicating if the string start and end with brackets and can
    be converted to a list.
    """
    if s.startswith('[') and s.endswith(']'):
        try:
            ast.literal_eval(s) 
            return True
        except ValueError:
            return False 
    else:
        return False
    
def str_islist(l_string):
    """
    Return a boolean indicating if the string contains brackets and can
    be converted to a list.
    """
    if isinstance(l_string, list):
        return [islist(string) for string in l_string]  
    else: 
        return islist(l_string)

def str_subset(l_string, pattern):
    if isinstance(l_string, list):
        return [string for string in l_string if re.search(pattern,string)]  
    else: 
        if re.search(pattern,l_string):
            return l_string
        else:
            return
        
def str_remove(l_string, pattern):
    """Remove strings matching the pattern."""
    if isinstance(l_string, list):
        return [string for string in l_string if not re.search(pattern,string)]  
    else: 
        if not re.search(pattern,l_string):
            return l_string
        else:
            return
        
def str_remove_empty(l_string):
    """Remove empty strings ''. """
    if isinstance(l_string, list):
        return [string for string in l_string if len(string) >= 1]  
    else: 
        if len(l_string) >=1:
            return l_string
        else:
            return
        
def str_locate(l_string, pattern):
    """Provide the index of strings matching the pattern"""
    if isinstance(l_string, list):
        return [i for i, string in enumerate(l_string) if re.search(pattern,string)]  
    else: 
       raise ValueError('Not implemented for str')

def str_collapse(l_string, sep=" "):
    """Collapse multiple strings into a single string"""
    if isinstance(l_string, list):
        return sep.join(l_string) 
    else: 
       return l_string

def str_detect(l_string, pattern): 
    """Return a boolean list indicating if pattern is found."""
    pattern = re.compile(pattern)
    if isinstance(l_string, list): 
        return [len(pattern.findall(s))>=1 for s in l_string]
    elif isinstance(l_string, str):
        return len(pattern.findall(l_string)) >=1
    else:
        return False
    
def str_extract(l_string, pattern, simplify=True):
    if isinstance(l_string, list):
        return [str_simplify(re.findall(pattern,string), simplify) for string in l_string]     
    else:
        return str_simplify(re.findall(pattern,l_string), simplify)   
    
def str_replace(l_string, pattern, replacement): 
    if isinstance(l_string, list):
        return [re.sub(pattern, replacement, string) for string in l_string]     
    else:
        return re.sub(pattern, replacement, l_string)
    
def str_sub(l_string, start=0, end=None): 
    if isinstance(l_string, list):
        return [string[start:end] for string in l_string]     
    else:
        return l_string[start:end]  
    
def str_pad(x, width, side="left", pad=" "):
    """
    Vectorised over string, width and pad.
    width: desired length of the string 
    pad: must be a single character
    """
    # Check input arguments 
    if (side not in ["left", "right", "both"]):
        raise ValueError("Valid side argument: left, right, both")
    # Check that pad are single characters 
    # TODO 
    # If x is a single string, adapt arguments 
    flag = "list"
    if isinstance(x, str):
        flag = "str"
        x = [x]
        width = [width]
        pad = [pad]
    # Vectorization of width and pad
    if flag == "list":
        width = rep(width, length=len(x))
        pad = rep(pad, length=len(x))
    # Padding 
    if (side == "left" or side == "both"): 
        x = [string.rjust(width[i], pad[i]) for i, string in enumerate(x)]  
    if side == "right":   
        x = [string.ljust(width[i], pad[i]) for i, string in enumerate(x)]  
    # Adapt output to x input type 
    if flag == "str":
        x = str_simplify(x)
    return x

def str_split(l_string, pattern, maxsplit=0): 
    """str_split(x, pattern) splits up a string into multiple pieces."""
    if isinstance(l_string, list):
        return [re.split(pattern, string, maxsplit=maxsplit) for string in l_string]     
    else:
        return re.split(pattern, l_string, maxsplit=maxsplit) 
    
# str_count(x, pattern) counts the number of patterns.

#-----------------------------------------------------------------------------.