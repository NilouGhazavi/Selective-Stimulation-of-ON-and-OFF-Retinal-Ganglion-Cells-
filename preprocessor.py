#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:08:47 2021

@author: saad
"""
import sys
from model.RiekeModel import RiekeModel, RiekeModel_tf
from model.data_handler import load_h5Dataset, save_h5Dataset, rolling_window, unroll_data
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import namedtuple 
Exptdata = namedtuple('Exptdata', ['X', 'y'])
import multiprocessing as mp
# from joblib import Parallel, delayed
import time
import gc
# from pyret.filtertools import sta, decompose
from scipy import signal
from scipy.signal import convolve
from scipy.special import gamma as scipy_gamma
from scipy.signal import lfilter
from scipy import integrate
import tensorflow as tf

#betaSlow makes things worse

def model_params_orig(timeBin=1):
# retina 1 
    ##  cones - monkey
    params_cones = {}
    params_cones['sigma'] =  22 #22  # rhodopsin activity decay rate (1/sec) - default 22
    params_cones['phi'] =  22     # phosphodiesterase activity decay rate (1/sec) - default 22
    params_cones['eta'] =  2000  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
    params_cones['k'] =  0.01     # constant relating cGMP to current - default 0.02
    params_cones['h'] =  3       # cooperativity for cGMP->current - default 3
    params_cones['cdark'] =  1  # dark calcium concentration - default 1
    params_cones['beta'] = 9 #16 # 9	  # rate constant for calcium removal in 1/sec - default 9
    params_cones['betaSlow'] =  0	  
    params_cones['hillcoef'] =  4 #4  	  # cooperativity for cyclase, hill coef - default 4
    params_cones['hillaffinity'] =  0.5   # hill affinity for cyclase - default 0.5
    params_cones['gamma'] =  10/timeBin #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_cones['timeStep'] =  1e-3  # freds default is 1e-4
    params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2

    ## rods - mice - ORIGINAL
    params_rods = {}
    params_rods['sigma'] = 7.66  # rhodopsin activity decay rate (1/sec) - default 22
    params_rods['phi'] =  7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
    params_rods['eta'] = 1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    params_rods['h'] =  3       # cooperativity for cGMP->current - default 3
    params_rods['cdark'] =  1#1  # dark calcium concentration - default 1
    params_rods['beta'] = 25	  # rate constant for calcium removal in 1/sec - default 9
    params_rods['betaSlow'] =  0	  
    params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    params_rods['gamma'] =  10/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2
    
    # rods - mice - Retina 1
    # params_rods = {}
    # params_rods['sigma'] = 9 #16 #30 # 7.66  # rhodopsin activity decay rate (1/sec) - default 22
    # params_rods['phi'] =  10 #16 #10 #7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_rods['eta'] = 4 #2.2 #1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    # params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    # params_rods['h'] =  3 #3       # cooperativity for cGMP->current - default 3
    # params_rods['cdark'] =  1  # dark calcium concentration - default 1
    # params_rods['beta'] =  10 #25	  # rate constant for calcium removal in 1/sec - default 9
    # params_rods['betaSlow'] =  0	  
    # params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    # params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    # params_rods['gamma'] =  800/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    # params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2

    # RETINA 2
    # # rods - mice - Retina 2
    # params_rods = {}
    # params_rods['sigma'] = 10 # 7.66  # rhodopsin activity decay rate (1/sec) - default 22
    # params_rods['phi'] =  11  #7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_rods['eta'] = 3  #1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    # params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    # params_rods['h'] =  3 #3       # cooperativity for cGMP->current - default 3
    # params_rods['cdark'] =  1  # dark calcium concentration - default 1
    # params_rods['beta'] =  10 #25	  # rate constant for calcium removal in 1/sec - default 9
    # params_rods['betaSlow'] =  0	  
    # params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    # params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    # params_rods['gamma'] =  850 #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    # params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2

    #     RETINA 3
    # # cones - retina3
    # params_cones = {}
    # params_cones['sigma'] =  22 #22  # rhodopsin activity decay rate (1/sec) - default 22
    # params_cones['phi'] =  22     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_cones['eta'] =  1600  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
    # params_cones['k'] =  0.01     # constant relating cGMP to current - default 0.02
    # params_cones['h'] =  3       # cooperativity for cGMP->current - default 3
    # params_cones['cdark'] =  1  # dark calcium concentration - default 1
    # params_cones['beta'] = 10 #16 # 9	  # rate constant for calcium removal in 1/sec - default 9
    # params_cones['betaSlow'] =  0	  
    # params_cones['hillcoef'] =  4 #4  	  # cooperativity for cyclase, hill coef - default 4
    # params_cones['hillaffinity'] =  0.5   # hill affinity for cyclase - default 0.5
    # params_cones['gamma'] =  5 #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_cones['timeStep'] =  1e-3  # freds default is 1e-4
    # params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2

    # rods - mice - Retina 3 (same as retina 1)
    # params_rods = {}
    # params_rods['sigma'] = 9 #9 # 7.66  # rhodopsin activity decay rate (1/sec) - default 22
    # params_rods['phi'] =  10 #10  #7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_rods['eta'] = 4  #1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    # params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    # params_rods['h'] =  3 #3       # cooperativity for cGMP->current - default 3
    # params_rods['cdark'] =  1  # dark calcium concentration - default 1
    # params_rods['beta'] =  10 #25	  # rate constant for calcium removal in 1/sec - default 9
    # params_rods['betaSlow'] =  0	  
    # params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    # params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    # params_rods['gamma'] =  800 #500 #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    # params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2
    
    return params_cones,params_rods
   
def model_params_monkey_orig(timeBin=1):
# retina 1 
    ##  cones - monkey
    params_cones = {}
    params_cones['sigma'] =  22 #22  # rhodopsin activity decay rate (1/sec) - default 22
    params_cones['phi'] =  22     # phosphodiesterase activity decay rate (1/sec) - default 22
    params_cones['eta'] =  2000  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
    params_cones['k'] =  0.01     # constant relating cGMP to current - default 0.02
    params_cones['h'] =  3       # cooperativity for cGMP->current - default 3
    params_cones['cdark'] =  1  # dark calcium concentration - default 1
    params_cones['beta'] = 9 #16 # 9	  # rate constant for calcium removal in 1/sec - default 9
    params_cones['betaSlow'] =  0	  
    params_cones['hillcoef'] =  4 #4  	  # cooperativity for cyclase, hill coef - default 4
    params_cones['hillaffinity'] =  0.5   # hill affinity for cyclase - default 0.5
    params_cones['gamma'] =  10/timeBin #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_cones['timeStep'] =  1e-3  # freds default is 1e-4
    params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2

    ## rods - mice - ORIGINAL
    params_rods = {}
    params_rods['sigma'] = 7.07  # rhodopsin activity decay rate (1/sec) - default 22
    params_rods['phi'] =  7.07     # phosphodiesterase activity decay rate (1/sec) - default 22
    params_rods['eta'] = 2.53	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_rods['gdark'] = 15.5 # 13.4 # concentration of cGMP in darkness - default 20.5
    params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    params_rods['h'] =  3       # cooperativity for cGMP->current - default 3
    params_rods['cdark'] =  1#1  # dark calcium concentration - default 1
    params_rods['beta'] = 25	  # rate constant for calcium removal in 1/sec - default 9
    params_rods['betaSlow'] =  0	  
    params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    params_rods['hillaffinity'] =  0.50		# affinity for Ca2+
    params_rods['gamma'] =  50/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2
    
    # rods - mice - Retina 1
    # params_rods = {}
    # params_rods['sigma'] = 9 #16 #30 # 7.66  # rhodopsin activity decay rate (1/sec) - default 22
    # params_rods['phi'] =  10 #16 #10 #7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_rods['eta'] = 4 #2.2 #1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    # params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    # params_rods['h'] =  3 #3       # cooperativity for cGMP->current - default 3
    # params_rods['cdark'] =  1  # dark calcium concentration - default 1
    # params_rods['beta'] =  10 #25	  # rate constant for calcium removal in 1/sec - default 9
    # params_rods['betaSlow'] =  0	  
    # params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    # params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    # params_rods['gamma'] =  800/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    # params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2

    # RETINA 2
    # # rods - mice - Retina 2
    # params_rods = {}
    # params_rods['sigma'] = 10 # 7.66  # rhodopsin activity decay rate (1/sec) - default 22
    # params_rods['phi'] =  11  #7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_rods['eta'] = 3  #1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    # params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    # params_rods['h'] =  3 #3       # cooperativity for cGMP->current - default 3
    # params_rods['cdark'] =  1  # dark calcium concentration - default 1
    # params_rods['beta'] =  10 #25	  # rate constant for calcium removal in 1/sec - default 9
    # params_rods['betaSlow'] =  0	  
    # params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    # params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    # params_rods['gamma'] =  850 #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    # params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2

    #     RETINA 3
    # # cones - retina3
    # params_cones = {}
    # params_cones['sigma'] =  22 #22  # rhodopsin activity decay rate (1/sec) - default 22
    # params_cones['phi'] =  22     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_cones['eta'] =  1600  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
    # params_cones['k'] =  0.01     # constant relating cGMP to current - default 0.02
    # params_cones['h'] =  3       # cooperativity for cGMP->current - default 3
    # params_cones['cdark'] =  1  # dark calcium concentration - default 1
    # params_cones['beta'] = 10 #16 # 9	  # rate constant for calcium removal in 1/sec - default 9
    # params_cones['betaSlow'] =  0	  
    # params_cones['hillcoef'] =  4 #4  	  # cooperativity for cyclase, hill coef - default 4
    # params_cones['hillaffinity'] =  0.5   # hill affinity for cyclase - default 0.5
    # params_cones['gamma'] =  5 #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_cones['timeStep'] =  1e-3  # freds default is 1e-4
    # params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2

    # rods - mice - Retina 3 (same as retina 1)
    # params_rods = {}
    # params_rods['sigma'] = 9 #9 # 7.66  # rhodopsin activity decay rate (1/sec) - default 22
    # params_rods['phi'] =  10 #10  #7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_rods['eta'] = 4  #1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    # params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    # params_rods['h'] =  3 #3       # cooperativity for cGMP->current - default 3
    # params_rods['cdark'] =  1  # dark calcium concentration - default 1
    # params_rods['beta'] =  10 #25	  # rate constant for calcium removal in 1/sec - default 9
    # params_rods['betaSlow'] =  0	  
    # params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    # params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    # params_rods['gamma'] =  800 #500 #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    # params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2
    
    return params_cones,params_rods
   

def model_params_monkey(timeBin=1):
# retina 1 
    ##  cones - monkey
    params_cones = {}
    params_cones['sigma'] =  22 #22  # rhodopsin activity decay rate (1/sec) - default 22
    params_cones['phi'] =  22     # phosphodiesterase activity decay rate (1/sec) - default 22
    params_cones['eta'] =  2000  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
    params_cones['k'] =  0.01     # constant relating cGMP to current - default 0.02
    params_cones['h'] =  3       # cooperativity for cGMP->current - default 3
    params_cones['cdark'] =  1  # dark calcium concentration - default 1
    params_cones['beta'] = 9 #16 # 9	  # rate constant for calcium removal in 1/sec - default 9
    params_cones['betaSlow'] =  0	  
    params_cones['hillcoef'] =  4 #4  	  # cooperativity for cyclase, hill coef - default 4
    params_cones['hillaffinity'] =  0.5   # hill affinity for cyclase - default 0.5
    params_cones['gamma'] =  10/timeBin #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_cones['timeStep'] =  1e-3  # freds default is 1e-4
    params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2

    ## rods - monkey - 0.3 rstar
    # params_rods = {}
    # params_rods['sigma'] = 25.8898  # rhodopsin activity decay rate (1/sec) - default 22
    # params_rods['phi'] =  21.1589     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_rods['eta'] = 76.97	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    # params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    # params_rods['h'] =  3       # cooperativity for cGMP->current - default 3
    # params_rods['cdark'] =  1#1  # dark calcium concentration - default 1
    # params_rods['beta'] = 47.38	  # rate constant for calcium removal in 1/sec - default 9
    # params_rods['betaSlow'] =  0	  
    # params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    # params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    # params_rods['gamma'] =  1048/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    # params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2
    
    ## rods - monkey - 3 rstar
    params_rods = {}
    params_rods['sigma'] = 10  # rhodopsin activity decay rate (1/sec) - default 22
    params_rods['phi'] =  10     # phosphodiesterase activity decay rate (1/sec) - default 22
    params_rods['eta'] = 20	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_rods['gdark'] = 20 # 13.4 # concentration of cGMP in darkness - default 20.5
    params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    params_rods['h'] =  3       # cooperativity for cGMP->current - default 3
    params_rods['cdark'] =  1#1  # dark calcium concentration - default 1
    params_rods['beta'] = 15	  # rate constant for calcium removal in 1/sec - default 9
    params_rods['betaSlow'] =  0	  
    params_rods['hillcoef'] =  5.2  	  # cooperativity for cyclase, hill coef - default 4
    params_rods['hillaffinity'] =  0.26		# affinity for Ca2+
    params_rods['gamma'] =  100/timeBin #143/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2
    

    ## rods - monkey - 30 rstar
    # params_rods = {}
    # params_rods['sigma'] = 25.8898  # rhodopsin activity decay rate (1/sec) - default 22
    # params_rods['phi'] =  21.1589     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_rods['eta'] = 76.97	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    # params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    # params_rods['h'] =  3       # cooperativity for cGMP->current - default 3
    # params_rods['cdark'] =  1#1  # dark calcium concentration - default 1
    # params_rods['beta'] = 47.38	  # rate constant for calcium removal in 1/sec - default 9
    # params_rods['betaSlow'] =  0	  
    # params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    # params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    # params_rods['gamma'] =  14/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    # params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2


    return params_cones,params_rods
   

def model_params_trainable(timeBin = 1):
    # Train most of cone params and then for rods, set initial params to cone params.
    ##  cones - trainable model - fewParams
    # /home/saad/data/analyses/data_kiersten/retina1/8ms_trainablePR/photopic-10000/PRFR_CNN2D_best
    params_cones = {}
    params_cones['sigma'] =  250 #22  # rhodopsin activity decay rate (1/sec) - default 22
    params_cones['phi'] =  40.7 #22    # phosphodiesterase activity decay rate (1/sec) - default 22
    params_cones['eta'] =  879  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
    params_cones['k'] =  0.01     # constant relating cGMP to current - default 0.02
    params_cones['h'] =  3       # cooperativity for cGMP->current - default 3
    params_cones['cdark'] =  1  # dark calcium concentration - default 1
    params_cones['beta'] = 110 # 9	  # rate constant for calcium removal in 1/sec - default 9
    params_cones['betaSlow'] =  10 #0	  
    params_cones['hillcoef'] =  2.64 #4  	  # cooperativity for cyclase, hill coef - default 4
    params_cones['hillaffinity'] =  1.51 #0.5   # hill affinity for cyclase - default 0.5
    params_cones['gamma'] =  10/timeBin #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_cones['timeStep'] =  1e-3  # freds default is 1e-4
    params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2
 
    # rods - mice - Retina 1 - trainable
    params_rods = {}
    params_rods['sigma'] = 10 #7.66  # rhodopsin activity decay rate (1/sec) 
    params_rods['phi'] =  10 #7.66     # phosphodiesterase activity decay rate (1/sec) 
    params_rods['eta'] = 20 #1.62	  # phosphodiesterase activation rate constant (1/sec) 
    params_rods['gdark'] = 20 # concentration of cGMP in darkness - default 20.5
    params_rods['k'] =  0.01    # constant relating cGMP to current - default 0.02
    params_rods['h'] =  3  # cooperativity for cGMP->current - default 3
    params_rods['cdark'] =  1  # dark calcium concentration - default 1
    params_rods['beta'] =  15.7 #25	  # rate constant for calcium removal in 1/sec - default 9
    params_rods['betaSlow'] =  0	  
    params_rods['hillcoef'] =  5.2 #13.8  	  # cooperativity for cyclase, hill coef - default 4
    params_rods['hillaffinity'] = 0.26 #0.22		# affinity for Ca2+
    params_rods['gamma'] =  2.5/timeBin #2.44/timeBin #2.44/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2


    return params_cones,params_rods

def model_params_prbp(timeBin = 1):
    # Retina 2
    # params from training pr-bp-cnns end to end
    params_cones = {}
    params_cones['sigma'] =  22 #22  # rhodopsin activity decay rate (1/sec) - default 22
    params_cones['phi'] =  22     # phosphodiesterase activity decay rate (1/sec) - default 22
    params_cones['eta'] =  2000  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
    params_cones['k'] =  0.01     # constant relating cGMP to current - default 0.02
    params_cones['h'] =  3       # cooperativity for cGMP->current - default 3
    params_cones['cdark'] =  1  # dark calcium concentration - default 1
    params_cones['beta'] = 9 #16 # 9	  # rate constant for calcium removal in 1/sec - default 9
    params_cones['betaSlow'] =  0	  
    params_cones['hillcoef'] =  4 #4  	  # cooperativity for cyclase, hill coef - default 4
    params_cones['hillaffinity'] =  0.5   # hill affinity for cyclase - default 0.5
    params_cones['gamma'] =  -0.90/timeBin #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_cones['timeStep'] =  1e-3  # freds default is 1e-4
    params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2
 
    # rods - mice - Retina 1 - trainable
    params_rods = {}
    params_rods['sigma'] = 11.36 #7.66  # rhodopsin activity decay rate (1/sec) 
    params_rods['phi'] =  11.32 #7.66     # phosphodiesterase activity decay rate (1/sec) 
    params_rods['eta'] = 44.7 #1.62	  # phosphodiesterase activation rate constant (1/sec) 
    params_rods['gdark'] = 28 # concentration of cGMP in darkness - default 20.5
    params_rods['k'] =  0.01    # constant relating cGMP to current - default 0.02
    params_rods['h'] =  3  # cooperativity for cGMP->current - default 3
    params_rods['cdark'] =  1  # dark calcium concentration - default 1
    params_rods['beta'] =  15.5 #25	  # rate constant for calcium removal in 1/sec - default 9
    params_rods['betaSlow'] =  0	  
    params_rods['hillcoef'] =  3.85 #13.8  	  # cooperativity for cyclase, hill coef - default 4
    params_rods['hillaffinity'] = 0.55 #0.22		# affinity for Ca2+
    params_rods['gamma'] =  32/timeBin #2.44/timeBin #2.44/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2


    return params_cones,params_rods



def parallel_runRiekeModel(params,stim_frames_photons,idx_pixelToTake):
    params['stm'] = stim_frames_photons[:,idx_pixelToTake]
    _,stim_currents = Model(params)
        
    return stim_currents

def parallel_runClarkModel(params,stim_photons,idx_pixelToTake):
    params['stm'] = stim_photons[:,idx_pixelToTake]
    _,stim_currents = DA_model_iter(params)
        
    return stim_currents

def run_model(pr_mdl_name,stim,resp,params,meanIntensity,upSampFac,downSampFac=17,n_discard=0,NORM=1,DOWN_SAMP=1,ROLLING_FAC=30,runOnGPU=False,ode_solver='RungeKutta',changeIntensities=False):
    
    stim_spatialDims = stim.shape[1:]
    stim = stim.reshape(stim.shape[0],stim.shape[1]*stim.shape[2])
    
    stim = np.repeat(stim,upSampFac,axis=0)
    
    if changeIntensities==True:
        stim[stim>0] = 2*meanIntensity
        stim[stim<0] = (2*meanIntensity)/300
        stim_photons = stim * params['timeStep']
        
    else:
        stim_photons = stim / upSampFac

    idx_allPixels = np.arange(0,stim.shape[1])
    
    # num_cores = mp.cpu_count()
    

    
    t = time.time()
    if pr_mdl_name == 'rieke':
        params['tme'] = np.arange(0,stim_photons.shape[0])*params['timeStep']
        params['biophysFlag'] = 1
        
        
        _,stim_currents = RiekeModel(params,stim_photons,ode_solver)
        
        
        if runOnGPU==True:
            stim_photons_tf = tf.convert_to_tensor(stim_photons,dtype=tf.float32)
            from model.RiekeModel import RiekeModel_tf
            _,stim_currents_tf = RiekeModel_tf(params,stim_photons_tf)
            
            # plt.plot(stim_currents_tf[:,0])
            # stim_currents_tf[0,0]
            stim_currents = np.array(stim_currents_tf)
        

        
    elif pr_mdl_name == 'clark':
        _,stim_currents = DA_model_iter(params,stim_photons)
        
        
    t_elasped_parallel = time.time()-t
    print('time elasped: '+str(t_elasped_parallel)+' seconds')
    

    # reshape back to spatial pixels and downsample

    if DOWN_SAMP == 1 and downSampFac>1:
        # 1
        # idx_downsamples = np.arange(0,stim_currents.shape[0],downSampFac)
        # stim_currents_downsampled = stim_currents[idx_downsamples]
        
        # 2
        # steps_downsamp = downSampFac
        # stim_currents_downsampled = stim_currents[steps_downsamp-1::steps_downsamp]
        
        # 3
        # stim_currents_downsampled = signal.resample(stim_currents,int(stim_currents.shape[0]/downSampFac))
        
        # 4
        rgb = stim_currents.T  
        rgb[np.isnan(rgb)] = np.nanmedian(rgb)
        
        rollingFac = ROLLING_FAC
        a = np.empty((len(idx_allPixels),rollingFac))
        a[:] = np.nan
        a = np.concatenate((a,rgb),axis=1)
        rgb8 = np.nanmean(rolling_window(a,rollingFac,time_axis = -1),axis=-1)
        rgb8 = rgb8.reshape(rgb8.shape[0],-1, downSampFac)    
        rgb8 = rgb8[:,:,0]
        
        
        stim_currents_downsampled = rgb8
        stim_currents_downsampled = stim_currents_downsampled.T
        


        
    else:
        stim_currents_downsampled = stim_currents
    
    stim_currents_reshaped = stim_currents_downsampled.reshape(stim_currents_downsampled.shape[0],stim_spatialDims[0],stim_spatialDims[1])
    stim_currents_reshaped = stim_currents_reshaped[n_discard:]
    
    if NORM==1:
        stim_currents_norm = (stim_currents_reshaped - np.min(stim_currents_reshaped)) / (np.max(stim_currents_reshaped)-np.min(stim_currents_reshaped))
        stim_currents_norm = stim_currents_norm - np.mean(stim_currents_norm)
    else:
        stim_currents_norm = stim_currents_reshaped
    
    # discard response if n_discard > 0
    if n_discard > 0:
        resp = resp[n_discard:]
        
    return stim_currents_norm,resp

def rwa_stim(X,y,temporal_window,idx_unit,t_start,t_end):
    
    stim = X[t_start:t_end,idx_unit,idx_unit]
    
    spikeRate =y[t_start:t_end,idx_unit,idx_unit]
    
    stim = rolling_window(stim,temporal_window)
    spikeRate = spikeRate[temporal_window:]
    rwa = np.nanmean(stim*spikeRate[:,None],axis=0)
    
    
    temporal_feature = rwa
    # plt.imshow(spatial_feature,cmap='winter')
    # plt.plot(temporal_feature)
    
    return temporal_feature

def DA_model(params):
    
    def generate_simple_filter(tau,n,t):
       f = (t**n)*np.exp(-t/tau); # functional form in paper
       f = (f/tau**(n+1))/scipy_gamma(n+1) # normalize appropriately
       return f
   
    def dxdt(t,x,params_ode):
        b = params_ode['beta'] 
        a = params_ode['alpha'] 
        tau_r = params_ode['tau_r'] 
        y = params_ode['y'] 
        z = params_ode['z'] 
        
        zt = np.interp(t,np.arange(0,z.shape[0]),z)
        yt = np.interp(t,np.arange(0,y.shape[0]),y)
        
        dx = (1/tau_r) * ((a*yt) - ((1+(b*zt))*x))
        
        return dx

    alpha = params['alpha'] / params['timeBin']
    beta = params['beta'] / params['timeBin']
    gamma = params['gamma'] / params['timeBin']
    tau_y = params['tau_y'] / params['timeBin']
    n_y = params['n_y']   
    tau_z = params['tau_z'] / params['timeBin']
    n_z = params['n_z']
    tau_r = params['tau_r'] / params['timeBin']
    
    stim = params['stm']
    
    t = np.arange(0,1000/timeBin)

    Ky = generate_simple_filter(tau_y,n_y,t)
    Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))

    y = lfilter(Ky,1,stim)
    z = lfilter(Kz,1,stim)
   
    if tau_r > 0:
        params_ode = {}
        params_ode['alpha'] = alpha
        params_ode['beta'] = beta
        params_ode['y'] = y
        params_ode['z'] = z
        params_ode['tau_r'] = tau_r
        
        T0 = np.array([1,z.shape[0]])
        X0 = 0
        
        ode15s = integrate.ode(dxdt).set_integrator('vode', method='bdf', order=5, max_step=25)
        ode15s.set_initial_value(X0).set_f_params(params_ode)
        dt = 10
        R = np.atleast_1d(0)
        T = np.atleast_1d(0)
        while ode15s.successful() and ode15s.t < T0[-1]:
            ode15s.integrate(ode15s.t+dt)
            R = np.append(R,ode15s.y)
            T = np.append(T,ode15s.t)        
        R = R[1:]
        T = T[1:]
        
        if dt>1:
            R = np.interp(np.arange(0,z.shape[0]),T,R)

    else:   
        R = alpha*y/(1+(beta*z))
        
        
    params['response'] = R
    
    return params,params['response']

def DA_model_iter(params,stim):
    
    def generate_simple_filter(tau,n,t):
       f = (t**n)*np.exp(-t/tau); # functional form in paper
       f = (f/tau**(n+1))/scipy_gamma(n+1) # normalize appropriately
       return f

    timeStep = params['timeStep']
    alpha = params['alpha']/timeStep
    beta = params['beta']/timeStep
    gamma = params['gamma']
    tau_y = params['tau_y']/timeStep
    n_y = params['n_y']   
    tau_z = params['tau_z']/timeStep
    n_z = params['n_z']
    tau_r = params['tau_r']/timeStep
    
        
    t = np.ceil(np.arange(0,1000/timeStep))

    Ky = generate_simple_filter(tau_y,n_y,t)
    Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))

    y = lfilter(Ky,1,stim,axis=0)
    z = lfilter(Kz,1,stim,axis=0)
   
    
    if tau_r > 0:
        
        zt = np.zeros((z.shape[0]))
        yt = np.zeros((z.shape[0]))
        R = np.zeros((z.shape[0]))
        
        zt[0] = z[0]
        yt[0] = y[0]
        R[0] = alpha*yt[0]/(1+(beta*zt[0]))
        
        for pnt in range(1,z.shape[0]):
            zt[pnt] = z[pnt] #np.interp(pnt,np.arange(0,z.shape[0]),z)
            yt[pnt] = y[pnt] #np.interp(pnt,np.arange(0,y.shape[0]),y)
            
            dx = (1/tau_r) * ((alpha*yt[pnt]) - ((1+(beta*zt[pnt]))*R[pnt-1]))
            R[pnt] = R[pnt-1] + dx
        
    else:   
        R = alpha*y/(1+(beta*z))
        
        
    params['response'] = R
    
    return params,params['response']

def model_params_clark():
    params_cones = {}
    # params_cones['alpha'] =  0.99
    # params_cones['beta'] = -0.02 
    # params_cones['gamma'] =  0.44 
    # params_cones['tau_y'] =  3.66 
    # params_cones['tau_z'] =  18.78 
    # params_cones['n_y'] =  11.84   
    # params_cones['n_z'] =  9.23 
    # params_cones['timeStep'] = 1e-3
    # params_cones['tau_r'] = 0
    
    # retina 1 - TRAINABLE
    params_cones = {}
    params_cones['alpha'] =  0.99
    params_cones['beta'] = -0.02 
    params_cones['gamma'] =  0.45
    params_cones['tau_y'] =  4
    params_cones['tau_z'] =  14.1 
    params_cones['n_y'] =  11.6
    params_cones['n_z'] =  13.8
    params_cones['timeStep'] = 1e-3
    params_cones['tau_r'] = 0


    # retina 1
    # params_rods = {}
    # params_rods['alpha'] =  1 
    # params_rods['beta'] =  0.9976
    # params_rods['gamma'] =  1.001
    # params_rods['tau_y'] =  14.98 
    # params_rods['tau_z'] = 10 
    # params_rods['n_y'] =  8.4554
    # params_rods['n_z'] =  10
    # params_rods['timeStep'] = 1e-3
    # params_rods['tau_r'] = 0 
    
    # retina 1 - from trainable
    # params_rods = {}
    # params_rods['alpha'] =  1 
    # params_rods['beta'] =  0.99
    # params_rods['gamma'] =  1
    # params_rods['tau_y'] =  14.98 
    # params_rods['tau_z'] = 10 
    # params_rods['n_y'] =  8.45
    # params_rods['n_z'] =  10 
    # params_rods['timeStep'] = 1e-3
    # params_rods['tau_r'] = 0 

    
    # retina 2
    # params_rods = {}
    # params_rods['alpha'] =  1  
    # params_rods['beta'] =  0.36 
    # params_rods['gamma'] =  0.448
    # params_rods['tau_y'] =  17
    # params_rods['n_y'] =  6.4
    # params_rods['tau_z'] =  1000
    # params_rods['n_z'] =  1
    # params_rods['timeStep'] = 1e-3
    # params_rods['tau_r'] = 0 #4.78
    
    # retina 2 - TRAINABLE
    params_rods = {}
    params_rods['alpha'] =  0.59  
    params_rods['beta'] =  4
    params_rods['gamma'] =  -9.22
    params_rods['tau_y'] =  13.7
    params_rods['n_y'] =  9.8
    params_rods['tau_z'] =  12.45
    params_rods['n_z'] =  2.7
    params_rods['timeStep'] = 1e-3
    params_rods['tau_r'] = 0 #4.78

 
    # retina 3
    # params_rods = {}
    # params_rods['alpha'] =  1  
    # params_rods['beta'] =  0.36 
    # params_rods['gamma'] =  0.448
    # params_rods['tau_y'] =  13
    # params_rods['n_y'] =  7.4
    # params_rods['tau_z'] =  166
    # params_rods['n_z'] =  1
    # params_rods['timeStep'] = 1e-3
    # params_rods['tau_r'] = 0 #4.78


    return params_cones,params_rods
    
def model_params_clark_init():
    params_cones = {}
    # params_cones['alpha'] =  0.99
    # params_cones['beta'] = -0.02 
    # params_cones['gamma'] =  0.44 
    # params_cones['tau_y'] =  3.66 
    # params_cones['tau_z'] =  18.78 
    # params_cones['n_y'] =  11.84   
    # params_cones['n_z'] =  9.23 
    # params_cones['timeStep'] = 1e-3
    # params_cones['tau_r'] = 0
    
    # retina 1 - TRAINABLE
    params_cones = {}
    params_cones['alpha'] =  1
    params_cones['beta'] = 1
    params_cones['gamma'] =  1
    params_cones['tau_y'] =  1
    params_cones['tau_z'] =  1 
    params_cones['n_y'] =  1
    params_cones['n_z'] =  1
    params_cones['timeStep'] = 1e-3
    params_cones['tau_r'] = 0


    # retina 1
    # params_rods = {}
    # params_rods['alpha'] =  1 
    # params_rods['beta'] =  0.9976
    # params_rods['gamma'] =  1.001
    # params_rods['tau_y'] =  14.98 
    # params_rods['tau_z'] = 10 
    # params_rods['n_y'] =  8.4554
    # params_rods['n_z'] =  10
    # params_rods['timeStep'] = 1e-3
    # params_rods['tau_r'] = 0 
    
    # retina 1 - from trainable
    # params_rods = {}
    # params_rods['alpha'] =  1 
    # params_rods['beta'] =  0.99
    # params_rods['gamma'] =  1
    # params_rods['tau_y'] =  14.98 
    # params_rods['tau_z'] = 10 
    # params_rods['n_y'] =  8.45
    # params_rods['n_z'] =  10 
    # params_rods['timeStep'] = 1e-3
    # params_rods['tau_r'] = 0 

    
    # retina 2
    # params_rods = {}
    # params_rods['alpha'] =  1  
    # params_rods['beta'] =  0.36 
    # params_rods['gamma'] =  0.448
    # params_rods['tau_y'] =  17
    # params_rods['n_y'] =  6.4
    # params_rods['tau_z'] =  1000
    # params_rods['n_z'] =  1
    # params_rods['timeStep'] = 1e-3
    # params_rods['tau_r'] = 0 #4.78
    
    # retina 2 - TRAINABLE
    params_rods = {}
    params_rods['alpha'] =  0.59  
    params_rods['beta'] =  4
    params_rods['gamma'] =  -9.22
    params_rods['tau_y'] =  13.7
    params_rods['n_y'] =  9.8
    params_rods['tau_z'] =  12.45
    params_rods['n_z'] =  2.7
    params_rods['timeStep'] = 1e-3
    params_rods['tau_r'] = 0 #4.78


    # retina 3
    # params_rods = {}
    # params_rods['alpha'] =  1  
    # params_rods['beta'] =  0.36 
    # params_rods['gamma'] =  0.448
    # params_rods['tau_y'] =  13
    # params_rods['n_y'] =  7.4
    # params_rods['tau_z'] =  166
    # params_rods['n_z'] =  1
    # params_rods['timeStep'] = 1e-3
    # params_rods['tau_r'] = 0 #4.78


    return params_cones,params_rods
    
        
# %% Single pr type

DEBUG_MODE = 1
WRITE_TO_H5 = 0
data_pers = 'kiersten'  #['saad','kiersten']
expDate = 'retina1' #'monkey01' #'retina1'  20180502_as3
lightLevel = 'photopic-10000' #'photopic-10000' #'photopic-10000'  # ['photopic','scotopic','mesopic-2026']
if expDate == 'monkey01':
    lightLevel = lightLevel+'-Rstar'
stimName = ''  # SACC_T2
folder = '8ms_sampShifted'  # 8ms_sampShifted
path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_'+data_pers+'/',expDate,'datasets/'+folder)
if stimName == '':
    fname_dataset = expDate+'_dataset_train_val_test_'+lightLevel+'.h5'
else:
    fname_dataset = expDate+'_dataset_train_val_test_'+stimName+'_'+lightLevel+'.h5'
fname_data_train_val_test = os.path.join(path_dataset,fname_dataset)


changeIntensities = False
pr_mdl_name = 'rieke'  # 'rieke' 'clark'
pr_type = 'cones'   # ['rods','cones']
ode_solver = 'Euler' #['hybrid','RungeKutta','Euler']

timeBin = 8
frameTime = 8
NORM = 1
NORM_FIXED = 0
DOWN_SAMP = 1
ROLLING_FAC = 2
upSampFac = int(frameTime/timeBin) #1#8 #17
downSampFac = upSampFac
meanIntensity = 0


if DOWN_SAMP==0:
    ROLLING_FAC = 0
else:
    ROLLING_FAC = 2


# t_frame = .008

if pr_mdl_name == 'rieke':
    params_cones,params_rods = model_params_trainable(timeBin)
    # params_cones,params_rods = model_params_monkey_orig(timeBin)
    params_cones['timeStep'] = 1e-3*(frameTime/upSampFac)
    params_rods['timeStep'] = 1e-3*(frameTime/upSampFac)
    

elif pr_mdl_name == 'clark':
    params_cones,params_rods = model_params_clark()
    params_cones['timeStep'] = frameTime/upSampFac
    params_rods['timeStep'] = frameTime/upSampFac



if pr_type == 'cones':
    params = params_cones
    y_lim = (-120,-40)
elif pr_type == 'rods':
    params = params_rods
    y_lim = (-10,2)


path_dataset_save = os.path.join(path_dataset)#,'filterTest')

if pr_mdl_name == 'rieke':
    dataset_name = lightLevel+'_mdl-'+pr_mdl_name+'_s-'+str(params['sigma'])+'_p-'+str(params['phi'])+'_e-'+str(params['eta'])+'_k-'+str(params['k'])+'_h-'+str(params['h'])+'_b-'+str(params['beta'])+'_hc-'+str(params['hillcoef'])+'_gd-'+str(params['gdark'])+'_g-'+str(params['gamma']*timeBin)+'_preproc-'+pr_type+'_norm-'+str(NORM)+'_tb-'+str(timeBin)+'_'+ode_solver+'_RF-'+str(ROLLING_FAC)
elif pr_mdl_name == 'clark':
    dataset_name = lightLevel+'_mdl-'+pr_mdl_name+'_a-'+str(params['alpha'])+pr_mdl_name+'_b-'+str(params['beta'])+'_g-'+str(params['gamma'])+'_y-'+str(params['tau_y'])+'_z-'+str(params['tau_z'])+'_r-'+str(params['tau_r'])+'_preproc-'+pr_type+'_norm-'+str(NORM)+'_rfac-'+str(ROLLING_FAC)+'_tb-'+str(timeBin)


fname_dataset_save = expDate+'_dataset_train_val_test_'+dataset_name+'.h5'

fname_dataset_save = os.path.join(path_dataset_save,fname_dataset_save)

if DEBUG_MODE==1:
    nsamps_end = 5000 #10000
    data_train_orig,data_val_orig,data_test,data_quality,dataset_rr,parameters,resp_orig = load_h5Dataset(fname_data_train_val_test,nsamps_val=nsamps_end*8/1000/60,nsamps_train=nsamps_end*8/1000/60)

else:
    nsamps_end = -1 
    data_train_orig,data_val_orig,data_test,data_quality,dataset_rr,parameters,resp_orig = load_h5Dataset(fname_data_train_val_test,nsamps_val=-1,nsamps_train=-1)


frames_X_orig = data_train_orig.X[:nsamps_end]

# Training data

stim_train,resp_train = run_model(pr_mdl_name,data_train_orig.X[:nsamps_end],data_train_orig.y[:nsamps_end],params,meanIntensity,upSampFac,downSampFac=downSampFac,n_discard=1000,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC,ode_solver=ode_solver,changeIntensities=changeIntensities)

if NORM==1:
    if NORM_FIXED==1:
        value_min = -40 #-113.211676833235 #tf.math.reduce_min(inputs)
        value_max = -28 #-81.1005869186533 #tf.math.reduce_max(inputs)
        stim_train_med = 0.57 # 0.43907212331830137 #tf.math.reduce_mean(R_norm)       
        
        stim_train_norm = (stim_train - value_min)/(value_max-value_min)
        stim_train_norm = stim_train_norm - stim_train_med
    else:
        value_min = np.min(stim_train)
        value_max = np.max(stim_train)
        stim_train_norm = (stim_train - value_min)/(value_max-value_min)
        stim_train_med = np.nanmean(stim_train_norm)
        stim_train_norm = stim_train_norm - stim_train_med
        
    parameters['value_min'] = value_min
    parameters['value_max'] = value_max
    parameters['stim_train_med'] = stim_train_med

else:
    stim_train_norm = stim_train


# Validation data
n_discard_val = 50
stim_val,resp_val = run_model(pr_mdl_name,data_val_orig.X,data_val_orig.y,params,meanIntensity,upSampFac,downSampFac=downSampFac,n_discard=n_discard_val,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC,runOnGPU=False,ode_solver=ode_solver)
if NORM==1:
    if NORM_FIXED!=1:
        value_min = np.min(stim_val)
        value_max = np.max(stim_val)

    stim_val_norm = (stim_val - value_min)/(value_max-value_min)
    # stim_val_med = np.nanmean(stim_val_norm)
    stim_val_norm = stim_val_norm - stim_train_med
    
else:
    stim_val_norm = stim_val 

# Update dataset
data_train = Exptdata(stim_train_norm,resp_train)
data_val = Exptdata(stim_val_norm,resp_val)
if data_pers == 'kiersten':
    dataset_rr['stim_0']['val'] = dataset_rr['stim_0']['val'][:,n_discard_val:,:]

# Update parameters
for j in params.keys():
    parameters[j] = params[j]
parameters['nsamps_end'] = nsamps_end


# plt.plot(stim_train[:,0,0])
# plt.ylim(y_lim)

# plt.plot(stim_train_norm[:,0,0])
# plt.plot(data_val_orig.X[n_discard_val:,0,0])
plt.plot(stim_val_norm[:,0,0])
# plt.plot(stim_val[:,0,0])
plt.title(ode_solver)
plt.show()

# RWA
if DOWN_SAMP==0:
    frames_X = np.repeat(frames_X_orig,upSampFac,axis=0)
    temporal_window = 60 * upSampFac
    # temporal_window = 1000
else:
    frames_X = frames_X_orig
    temporal_window = 60*2 # 60

n_discard = frames_X.shape[0]-stim_train_norm.shape[0]
frames_X = frames_X[n_discard:]

frames_X[frames_X>0] = 2*meanIntensity
frames_X[frames_X<0] = (2*meanIntensity)/300
# frames_X = frames_X / params['timeStep']  

idx_unit = 5
t_start = 0
t_end = stim_train_norm.shape[0]

# temporal_feature = rwa_stim(frames_X,stim_train_norm,temporal_window,idx_unit,t_start,t_end)
# plt.plot(filt_rieke,'k')
# plt.plot(temporal_feature)
# plt.title(dataset_name)
# plt.show()

# rgb = np.where(temporal_feature==np.max(temporal_feature))
# print(temporal_window-rgb[0][0])
# print(((temporal_window-rgb[0][0])*8)-22)
# print(rgb)

# Save dataset
# if DEBUG_MODE==0 and WRITE_TO_H5==1:
if WRITE_TO_H5==1:
    save_h5Dataset(fname_dataset_save,data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=resp_orig)


# a = stim_val_norm
# plt.plot(b[:,0,0])
# plt.plot(a[:,0,0])
# plt.show()

# a_tf = temporal_feature
# plt.plot(b_tf)
# plt.plot(a_tf)
# plt.show()


# %% Added pr signals

DEBUG_MODE = 0
WRITE_TO_H5 = 0
pr_mdl_name = 'rieke'  # 'rieke' 'clark'
expDate = 'retina1'
lightLevels = ('scotopic','photopic',)  # ['scotopic','photopic']
pr_type = ('rods','cones')   # ['rods','cones']
ode_solver = 'RungeKutta' #['hybrid','RungeKutta','Euler']
folder = '8ms_sampShifted'
timeBin = 4
frameTime = 8
NORM = 1
DOWN_SAMP = 1
ROLLING_FAC = 2
upSampFac = int(frameTime/timeBin) #1#8 #17
downSampFac = upSampFac

if DOWN_SAMP==0:
    ROLLING_FAC = 0
else:
    ROLLING_FAC = 2


if pr_mdl_name == 'rieke':
    params_cones,params_rods = model_params_t3(timeBin)
    params_cones['timeStep'] = 1e-3*(frameTime/upSampFac)
    params_rods['timeStep'] = 1e-3*(frameTime/upSampFac)
    thresh_lower_cones = -params_cones['darkCurrent'] - 10
    thresh_lower_rods = -params_rods['darkCurrent']-10

elif pr_mdl_name == 'clark':
    params_cones,params_rods = model_params_clark()
    params_cones['timeStep'] = frameTime/upSampFac
    params_rods['timeStep'] = frameTime/upSampFac
    thresh_lower_cones = -1000
    thresh_lower_rods = -1000



meanIntensities = {
    'scotopic': 1,
    'photopic': 10000
    }



path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/',expDate,'datasets/'+folder)

for l in lightLevels:
    fname_dataset = expDate+'_dataset_train_val_test_'+l+'.h5'
    fname_data_train_val_test = os.path.join(path_dataset,fname_dataset)
    
    
    
    data_train_orig,data_val_orig,data_test,data_quality,dataset_rr,parameters,resp_orig = load_h5Dataset(fname_data_train_val_test)
    
    if DEBUG_MODE==1:
        nsamps_end = 6000  #10000
    else:
        nsamps_end = data_train_orig.X.shape[0]-1 

    frames_X_orig = data_train_orig.X[:nsamps_end]

    
    fac_med = 5
    
    params=params_rods
    if pr_mdl_name == 'rieke':
        dataset_name = l+'-'+str(meanIntensities[l])+'_mdl-'+pr_mdl_name+'_s-'+str(params['sigma'])+'_p-'+str(params['phi'])+'_e-'+str(params['eta'])+'_g-'+str(params['gamma'])+'_k-'+str(params['k'])+'_h-'+str(params['h'])+'_b-'+str(params['beta'])+'_hc-'+str(params['hillcoef'])+'_gd-'+str(params['gdark'])+'_preproc-added'+'_norm-'+str(NORM)+'_tb-'+str(timeBin)+'_'+ode_solver+'_RF-'+str(ROLLING_FAC)
    elif pr_mdl_name == 'clark':
        dataset_name = l+'-'+str(meanIntensities[l])+'_mdl-'+pr_mdl_name+'_b-'+str(params['beta'])+'_g-'+str(params['gamma'])+'_y-'+str(params['tau_y'])+'_z-'+str(params['tau_z'])+'_preproc-added'+'_norm-'+str(NORM)
        
    
    fname_dataset_save = expDate+'_dataset_train_val_test_'+dataset_name+'.h5'
    fname_dataset_save = os.path.join(path_dataset,fname_dataset_save)


    
# Training data
    rods_stim_train,resp_train = run_model(pr_mdl_name,data_train_orig.X[:nsamps_end],data_train_orig.y[:nsamps_end],params_rods,meanIntensities[l],upSampFac,downSampFac=downSampFac,n_discard=1000,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC,ode_solver=ode_solver)
    cones_stim_train,resp_train = run_model(pr_mdl_name,data_train_orig.X[:nsamps_end],data_train_orig.y[:nsamps_end],params_cones,meanIntensities[l],upSampFac,downSampFac=downSampFac,n_discard=1000,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC,ode_solver=ode_solver)

    # rods_stim_train,resp_train = run_model(data_train.X,data_train.y,params_rods,meanIntensities[l],upSampFac,n_discard=1000,NORM=0)  
    # cones_stim_train,resp_train = run_model(data_train.X,data_train.y,params_cones,meanIntensities[l],upSampFac,n_discard=1000,NORM=0)
    
    rods_stim_train[np.isnan(rods_stim_train)] = 0
    
    med_rods = np.median(rods_stim_train)
    if med_rods >-1:
        med_thresh = -1
    else:
        med_thresh = fac_med * med_rods

    idx_discard_rods_1 = np.any(np.any(rods_stim_train>2,axis=1),axis=1)
    idx_discard_rods_2 = np.any(np.any(rods_stim_train<thresh_lower_rods,axis=1),axis=1)       
    idx_discard_rods_3 = np.any(np.any(rods_stim_train<med_thresh,axis=1),axis=1)
    idx_discard_rods = np.logical_or(idx_discard_rods_1,idx_discard_rods_2)
    idx_discard_rods = np.logical_or(idx_discard_rods,idx_discard_rods_3)
    idx_toTake_rods = np.where(idx_discard_rods==False)[0]
    
    idx_discard_cones_1 = np.any(np.any(cones_stim_train>10,axis=1),axis=1)
    idx_discard_cones_2 = np.any(np.any(cones_stim_train<thresh_lower_cones,axis=1),axis=1)
    idx_discard_cones = np.logical_or(idx_discard_cones_1,idx_discard_cones_2)
    idx_toTake_cones = np.where(idx_discard_cones==False)[0]   
    
    idx_discard = np.logical_or(idx_discard_rods,idx_discard_cones)
    idx_toTake_train = np.where(idx_discard==False)[0]
    
    # rods_stim_train_toTake = rods_stim_train[idx_toTake_train]
    rods_stim_train_toTake = rods_stim_train
    rods_stim_train_toTake[idx_discard_rods] = np.median(rods_stim_train_toTake[idx_toTake_rods])
    
    # cones_stim_train_toTake = cones_stim_train[idx_toTake_train]
    cones_stim_train_toTake = cones_stim_train
    cones_stim_train_toTake[idx_discard_cones] = np.median(cones_stim_train_toTake[idx_toTake_cones])

    # stim_train_added = rods_stim_train[idx_toTake] + cones_stim_train[idx_toTake]
    # stim_train_norm = stim_train_added
    
    if NORM==1:
        # value_min = np.percentile(stim_train_norm,0)
        # value_max = np.percentile(stim_train_norm,100)
        
        # stim_train_norm = (stim_train_norm - value_min)/(value_max-value_min)
        # value_median = np.median(stim_train_norm)
        
        # stim_train_norm = stim_train_norm - value_median
        
        rods_stim_train_clean = rods_stim_train_toTake
        rods_stim_train_med = np.median(rods_stim_train_clean)
        rods_stim_train_norm = rods_stim_train_clean - rods_stim_train_med
        
        cones_stim_train_clean = cones_stim_train_toTake
        cones_stim_train_med = np.median(cones_stim_train_clean)
        cones_stim_train_norm = cones_stim_train_clean - cones_stim_train_med
        
        stim_train_norm = rods_stim_train_norm + cones_stim_train_norm

        
        value_min = np.percentile(stim_train_norm,0)
        value_max = np.percentile(stim_train_norm,100)
        
        stim_train_norm = (stim_train_norm - value_min)/(value_max-value_min)
        stim_train_norm_mean = np.mean(stim_train_norm)
        stim_train_norm = stim_train_norm - stim_train_norm_mean
        
    elif NORM == 'medSub':
        rods_stim_train_clean = rods_stim_train[idx_toTake_train]
        rods_stim_train_med = np.median(rods_stim_train_clean)
        rods_stim_train_norm = rods_stim_train_clean - rods_stim_train_med
        
        cones_stim_train_clean = cones_stim_train[idx_toTake_train]
        cones_stim_train_med = np.median(cones_stim_train_clean)
        cones_stim_train_norm = cones_stim_train_clean - cones_stim_train_med
        
        stim_train_norm = rods_stim_train_norm + cones_stim_train_norm
        
  
    
    # plt.plot(stim_train_norm[:,0,0])
    
# Validation data
    n_discard_val = 50
    # run_model(stim,resp,params,meanIntensity,upSampFac,downSampFac=17,n_discard=0,NORM=1,DOWN_SAMP=1,ROLLING_FAC=30):
    rods_stim_val,resp_val = run_model(pr_mdl_name,data_val_orig.X,data_val_orig.y,params_rods,meanIntensities[l],upSampFac,downSampFac=downSampFac,n_discard=n_discard_val,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC,ode_solver=ode_solver)
    cones_stim_val,_ = run_model(pr_mdl_name,data_val_orig.X,data_val_orig.y,params_cones,meanIntensities[l],upSampFac,downSampFac=downSampFac,n_discard=n_discard_val,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC,ode_solver=ode_solver)

    
    idx_discard_rods_1 = np.any(np.any(rods_stim_val>1,axis=1),axis=1)
    idx_discard_rods_2 = np.any(np.any(rods_stim_val<thresh_lower_rods,axis=1),axis=1)
    idx_discard_rods_3 = np.any(np.any(rods_stim_val<med_thresh,axis=1),axis=1)
    idx_discard_rods = np.logical_or(idx_discard_rods_1,idx_discard_rods_2)
    idx_discard_rods = np.logical_or(idx_discard_rods,idx_discard_rods_3)   
    idx_toTake_rods = np.where(idx_discard_rods==False)[0]
    
    idx_discard_cones_1 = np.any(np.any(cones_stim_val>1,axis=1),axis=1)
    idx_discard_cones_2 = np.any(np.any(cones_stim_val<thresh_lower_cones,axis=1),axis=1)
    idx_discard_cones = np.logical_or(idx_discard_cones_1,idx_discard_cones_2)
    idx_toTake_cones = np.where(idx_discard_cones==False)[0]   
    
    idx_discard = np.logical_or(idx_discard_rods,idx_discard_cones)
    idx_toTake_val = np.where(idx_discard==False)[0]
    
    rods_stim_val_toTake = rods_stim_val.copy()
    rods_stim_val_toTake[idx_discard_rods] = np.nanmedian(rods_stim_val_toTake[idx_toTake_rods])
    
    cones_stim_val_toTake = cones_stim_val.copy()
    cones_stim_val_toTake[idx_discard_cones] = np.median(cones_stim_val_toTake[idx_toTake_cones])
    # rods_stim_val_toTake = rods_stim_val[idx_toTake_val] 
    # cones_stim_val_toTake = cones_stim_val[idx_toTake_val]
    stim_val_added = rods_stim_val_toTake + cones_stim_val_toTake
    # stim_val_added = cones_stim_val   


    stim_val_norm = stim_val_added
    
    if NORM==1:
        # stim_val_norm = (stim_val_norm - value_min)/(value_max-value_min)
        # stim_val_norm = stim_val_norm - value_median
        
        rods_stim_val_clean = rods_stim_val_toTake
        rods_stim_val_norm = rods_stim_val_clean - rods_stim_train_med
        
        cones_stim_val_clean = cones_stim_val_toTake
        cones_stim_val_norm = cones_stim_val_clean - cones_stim_train_med
        
        stim_val_norm = rods_stim_val_norm + cones_stim_val_norm

        stim_val_norm = (stim_val_norm - value_min)/(value_max-value_min)
        stim_val_norm = stim_val_norm - stim_train_norm_mean

    elif NORM == 'medSub':
        rods_stim_val_clean = rods_stim_val_toTake
        rods_stim_val_norm = rods_stim_val_clean - rods_stim_train_med
        
        cones_stim_val_clean = cones_stim_val_toTake
        cones_stim_val_norm = cones_stim_val_clean - cones_stim_train_med
        
        stim_val_norm = rods_stim_val_norm + cones_stim_val_norm


    # update datasets with new values
    
    # resp_train = resp_train[idx_toTake_train]
    data_train = Exptdata(stim_train_norm,resp_train)


    # resp_val = resp_val[idx_toTake_val]
    data_val = Exptdata(stim_val_norm,resp_val)
    
    dataset_rr['stim_0']['val'] = dataset_rr['stim_0']['val'][:,n_discard_val:,:]
    # dataset_rr['stim_0']['val'] = dataset_rr['stim_0']['val'][:,idx_toTake_val,:]
    
    if l == 'photopic':
        params = params_cones
    elif l== 'scotopic':
        params = params_rods
        
    for j in params.keys():
        parameters[j] = params[j]
    # parameters['meanIntensities'] = meanIntensities
    # parameters['pr_type'] = pr_type
    
    # Save dataset
    if WRITE_TO_H5==1:
        save_h5Dataset(fname_dataset_save,data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=resp_orig)
   
   
    # RWA
    if DOWN_SAMP==0:
        frames_X = np.repeat(frames_X_orig,upSampFac,axis=0)
        temporal_window = 60 * upSampFac
        temporal_window = 1000
    else:
        frames_X = frames_X_orig
        temporal_window = 60*2 # 60
    
    n_discard = frames_X.shape[0]-stim_train_norm.shape[0]
    frames_X = frames_X[n_discard:]
    
    frames_X[frames_X>0] = 2*meanIntensities[l]
    frames_X[frames_X<0] = (2*meanIntensities[l])/300
    # frames_X = frames_X / params['timeStep']  
    
    idx_unit = 5
    t_start = 0
    t_end = stim_train_norm.shape[0]
    
    temporal_feature = rwa_stim(frames_X,stim_train_norm,temporal_window,idx_unit,t_start,t_end)
    # plt.plot(temporal_feature)
    # plt.title(dataset_name)
    
    rgb = np.where(temporal_feature==np.max(temporal_feature))
    # print(temporal_window-rgb[0][0])
    # print(((temporal_window-rgb[0][0])*8)-22)
    
# b = stim_val_norm
# c = a-b
# plt.plot(stim_train_norm[:,0,0])
# plt.ylim((-120,2))
# plt.plot(stim_val_norm[:,0,0])
plt.plot(rods_stim_val[:,0,0])
plt.plot(cones_stim_val[:,0,0])
plt.title(ode_solver)
plt.show()

# plt.plot(c[:,0,0])

# plt.ylim((-120,2))
# plt.plot(a)
# plt.plot(b)
# plt.show()


# %% plotting
t_frame = 8
n_samps = 200
t_axis = np.arange(0,n_samps*t_frame,t_frame)
fontsize_ticks = 16
fontsize_labels = 16

figs,axs = plt.subplots(2,1,figsize=(5,11))

axs[0].plot(t_axis,cones_stim_val[:n_samps,0,0])
axs[0].set_xlabel('Time (ms)',fontsize=fontsize_labels)
axs[0].set_ylabel('Photocurrent (pA)',fontsize=fontsize_labels)
axs[0].tick_params(axis='both',labelsize=fontsize_ticks)
axs[0].set_title('cones')
axs[0].set_ylim((-150,-35))

axs[1].plot(t_axis,rods_stim_val[:n_samps,0,0])
axs[1].set_xlabel('Time (ms)',fontsize=fontsize_labels)
axs[1].set_ylabel('Photocurrent (pA)',fontsize=fontsize_labels)
axs[1].tick_params(axis='both',labelsize=fontsize_ticks)
axs[1].set_title('rods')

figs,axs = plt.subplots(2,1,figsize=(5,11))

axs[0].plot(t_axis,stim_val_norm[:n_samps,0,0])
axs[0].set_xlabel('Time (ms)',fontsize=fontsize_labels)
axs[0].set_ylabel('Photocurrent (pA)',fontsize=fontsize_labels)
axs[0].tick_params(axis='both',labelsize=fontsize_ticks)
axs[0].set_title('Added and normalized')

meanIntensity = 1
stim = np.array(data_val_orig.X)
stim = stim.reshape(stim.shape[0],stim.shape[1]*stim.shape[2])
stim[stim>0] = 2*meanIntensity
stim[stim<0] = (2*meanIntensity)/300
axs[1].plot(t_axis,stim[:n_samps,1])
axs[1].set_xlabel('Time (ms)',fontsize=fontsize_labels)
axs[1].set_ylabel('R*/rod/sec',fontsize=fontsize_labels)
axs[1].tick_params(axis='both',labelsize=fontsize_ticks)
axs[1].set_title('Stimulus')


# plt.setp(axs,aspect='equal')

# %% plotting
temp_cones = temp_feat_cone/temp_feat_cone.max()
temp_rods = temp_feat_scot/temp_feat_scot.max()
temp_rods_orig = temp_feat_scot_orig/temp_feat_scot_orig.max()
temp_rods_orig_g800 = temp_feat_scot_orig_g800/temp_feat_scot_orig_g800.max()

temp_cones = np.flip(temp_cones)
temp_rods = np.flip(temp_rods)
temp_rods_orig = np.flip(temp_rods_orig)
temp_rods_orig_g800 = np.flip(temp_rods_orig_g800)

temp_rods_orig_diff = np.diff(temp_rods_orig)
temp_rods_orig_diff = temp_rods_orig_diff/temp_rods_orig_diff.max()

temp_cones_diff = np.diff(temp_cones)
temp_cones_diff = temp_cones_diff/temp_cones_diff.max()


fig,axs = plt.subplots(1,1,figsize=(10,5))
t_axis = np.arange(0,240*8,8)
t_axis_ticks = np.arange(240*8,0,-40)
axs.plot(t_axis,temp_cones,color='gray',label='cone')
# axs.plot(t_axis[:-1],temp_cones_diff,'--',color='gray',label='cone_diff')
axs.plot(t_axis,temp_rods_orig,color='black',label='rod_orig')
# axs.plot(t_axis[:-1],temp_rods_orig_diff,'--',color='black',label='rod_orig_deriv')
axs.plot(t_axis,temp_rods_orig_g800,color='blue',label='rod_orig_g800')
axs.plot(t_axis,temp_rods,color='red',label='rod_model')
axs.set_xticks(t_axis[::10])
# axs.set_xticklabels(t_axis_ticks)
axs.set_xlabel('Time (ms)')
axs.set_ylabel('Normalized stim')
axs.legend(loc='best')
axs.set_xlim((0,1500))

# pr_filts = {
#     'temp_feat_cone': temp_feat_cone,
#     'temp_feat_rod': temp_feat_scot,
#     'temp_feat_rod_orig': temp_feat_scot_orig,
#     'temp_feat_rod_orig_g800': temp_feat_scot_orig_g800,
#     't_axis': t_axis,
#     'samp_rate': upSampFac
#     }
    
# import h5py
# fname = os.path.join(path_dataset_save,'pr_filters.h5')
# f = h5py.File(fname,'w')
# for i in list(pr_filts.keys()):
#     f.create_dataset(i, data=np.atleast_1d(pr_filts[i]),compression='gzip') 
# f.close()

# %% Seperate channels signals

expDate = 'retina3'
lightLevels = ('photopic',)  # ['scotopic','photopic']
pr_type = ('rods','cones')   # ['rods','cones']

meanIntensities = {
    'scotopic': 1,
    'photopic': 10000
    }


t_frame = 0.017
upSampFac = 17
NORM = 'medSub'

path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/',expDate,'datasets/temp')

for l in lightLevels:
    fname_dataset = expDate+'_dataset_train_val_test_'+l+'.h5'
    fname_data_train_val_test = os.path.join(path_dataset,fname_dataset)
    
    fname_dataset_save = expDate+'_dataset_train_val_test_'+l+'_preproc_chans_norm_'+str(NORM)+'.h5'
    fname_dataset_save = os.path.join(path_dataset,fname_dataset_save)

    data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig = load_h5Dataset(fname_data_train_val_test)
    
    params_cones,params_rods = model_params()
    thresh_lower = -params_cones['darkCurrent'] -  params_rods['darkCurrent'] - 10
    
# Training data
    rods_stim_train,resp_train = run_model(data_train.X,data_train.y,params_rods,meanIntensities[l],upSampFac,n_discard=100,NORM=0)
    cones_stim_train,resp_train = run_model(data_train.X,data_train.y,params_cones,meanIntensities[l],upSampFac,n_discard=100,NORM=0)
    
    idx_discard_rods_1 = np.any(np.any(rods_stim_train>10,axis=1),axis=1)
    idx_discard_rods_2 = np.any(np.any(rods_stim_train<-20,axis=1),axis=1)
    idx_discard_rods = np.logical_or(idx_discard_rods_1,idx_discard_rods_2)
    idx_toTake_rods = np.where(idx_discard_rods==False)[0]
    
    idx_discard_cones_1 = np.any(np.any(cones_stim_train>10,axis=1),axis=1)
    idx_discard_cones_2 = np.any(np.any(cones_stim_train<thresh_lower,axis=1),axis=1)
    idx_discard_cones = np.logical_or(idx_discard_cones_1,idx_discard_cones_2)
    idx_toTake_cones = np.where(idx_discard_cones==False)[0]   
    
    idx_discard = np.logical_or(idx_discard_rods,idx_discard_cones)
    idx_toTake = np.where(idx_discard==False)[0]
      
    
    stim_train_added = rods_stim_train[idx_toTake] + cones_stim_train[idx_toTake]
    stim_train_norm = stim_train_added
    
    if NORM==1:
        # value_min = np.percentile(stim_train_norm,0)
        # value_max = np.percentile(stim_train_norm,100)
        
        # stim_train_norm = (stim_train_norm - value_min)/(value_max-value_min)
        # value_median = np.median(stim_train_norm)
        
        # stim_train_norm = stim_train_norm - value_median
        
        rods_stim_train_clean = rods_stim_train[idx_toTake]
        rods_stim_train_med = np.median(rods_stim_train_clean)
        rods_stim_train_norm = rods_stim_train_clean - rods_stim_train_med
        
        cones_stim_train_clean = cones_stim_train[idx_toTake]
        cones_stim_train_med = np.median(cones_stim_train_clean)
        cones_stim_train_norm = cones_stim_train_clean - cones_stim_train_med
        
        stim_train_norm = rods_stim_train_norm + cones_stim_train_norm

        
        value_min = np.percentile(stim_train_norm,0)
        value_max = np.percentile(stim_train_norm,100)
        
        stim_train_norm = (stim_train_norm - value_min)/(value_max-value_min)
        stim_train_norm_mean = np.mean(stim_train_norm)
        stim_train_norm = stim_train_norm_mean - stim_train_norm_mean

        
    elif NORM == 'medSub':
        rods_stim_train_clean = rods_stim_train[idx_toTake]
        rods_stim_train_med = np.median(rods_stim_train_clean)
        rods_stim_train_norm = rods_stim_train_clean - rods_stim_train_med
        
        cones_stim_train_clean = cones_stim_train[idx_toTake]
        cones_stim_train_med = np.median(cones_stim_train_clean)
        cones_stim_train_norm = cones_stim_train_clean - cones_stim_train_med
        
        stim_train_norm = rods_stim_train_norm + cones_stim_train_norm
        
  
    resp_train = resp_train[idx_toTake]
    data_train = Exptdata(stim_train_norm,resp_train)
    
    plt.plot(rods_stim_train_norm[:,0,0])
    plt.ylim((-1,+1))
    
# Validation data
    n_discard_val = 20
    rods_stim_val,resp_val = run_model(data_val.X,data_val.y,params_rods,meanIntensities[l],upSampFac,n_discard=n_discard_val,NORM=0)
    cones_stim_val,_ = run_model(data_val.X,data_val.y,params_cones,meanIntensities[l],upSampFac,n_discard=n_discard_val,NORM=0)
    
    idx_discard_rods_1 = np.any(np.any(rods_stim_val>1,axis=1),axis=1)
    idx_discard_rods_2 = np.any(np.any(rods_stim_val<-100,axis=1),axis=1)
    idx_discard_rods = np.logical_or(idx_discard_rods_1,idx_discard_rods_2)
    idx_toTake_rods = np.where(idx_discard_rods==False)[0]
    
    idx_discard_cones_1 = np.any(np.any(cones_stim_val>1,axis=1),axis=1)
    idx_discard_cones_2 = np.any(np.any(cones_stim_val<thresh_lower,axis=1),axis=1)
    idx_discard_cones = np.logical_or(idx_discard_cones_1,idx_discard_cones_2)
    idx_toTake_cones = np.where(idx_discard_cones==False)[0]   
    
    idx_discard = np.logical_or(idx_discard_rods,idx_discard_cones)
    idx_toTake = np.where(idx_discard==False)[0]
    
    stim_val_added = rods_stim_val[idx_toTake] + cones_stim_val[idx_toTake]
    # stim_val_added = cones_stim_val   

    stim_val_norm = stim_val_added
    if NORM==1:
        # stim_val_norm = (stim_val_norm - value_min)/(value_max-value_min)
        # stim_val_norm = stim_val_norm - value_median
        
        rods_stim_val_clean = rods_stim_val[idx_toTake]
        rods_stim_val_norm = rods_stim_val_clean - rods_stim_train_med
        
        cones_stim_val_clean = cones_stim_val[idx_toTake]
        cones_stim_val_norm = cones_stim_val_clean - cones_stim_train_med
        
        stim_val_norm = rods_stim_val_norm + cones_stim_val_norm

        stim_val_norm = (stim_val_norm - value_min)/(value_max-value_min)
        stim_val_norm = stim_val_norm - stim_train_norm_mean


    elif NORM == 'medSub':
        rods_stim_val_clean = rods_stim_val[idx_toTake]
        rods_stim_val_norm = rods_stim_val_clean - rods_stim_train_med
        
        cones_stim_val_clean = cones_stim_val[idx_toTake]
        cones_stim_val_norm = cones_stim_val_clean - cones_stim_train_med
        
        stim_val_norm = rods_stim_val_norm + cones_stim_val_norm


    resp_val = resp_val[idx_toTake]
    data_val = Exptdata(stim_val_norm,resp_val)
    
    dataset_rr['stim_0']['val'] = dataset_rr['stim_0']['val'][:,n_discard_val:,:]
    dataset_rr['stim_0']['val'] = dataset_rr['stim_0']['val'][:,idx_toTake,:]
        
    
    # parameters['meanIntensities'] = meanIntensities
    # parameters['pr_type'] = pr_type
    
    
    save_h5Dataset(fname_dataset_save,data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=resp_orig)
    
plt.plot(stim_val_norm[:,0,0])
plt.ylim((-50,+50))




# %% Norm values
DEBUG_MODE=0
WRITE_TO_H5 = 0
pr_mdl_name = 'clark'  # 'rieke' 'clark'
expDate = 'retina1'
lightLevel = 'photopic-10000_mdl-rieke_s-22_p-22_e-2000_k-0.01_h-3_b-9_hc-4_gd-28_preproc-cones_norm-0_tb-4_Euler_RF-2'  # ['photopic','scotopic']
pr_type = 'cones'   # ['rods','cones']
ode_solver = 'Euler' #['hybrid','RungeKutta','Euler']
folder = '8ms_sampShifted'
timeBin = 8
frameTime = 8
NORM = 1
DOWN_SAMP = 1
ROLLING_FAC = 2
upSampFac = int(frameTime/timeBin) #1#8 #17
downSampFac = upSampFac

if DOWN_SAMP==0:
    ROLLING_FAC = 0
else:
    ROLLING_FAC = 2


if lightLevel == 'scotopic':
    meanIntensity = 1
elif lightLevel == 'photopic':
    meanIntensity = 10000


# t_frame = .008

if pr_mdl_name == 'rieke':
    # params_cones,params_rods = model_params_trainable(timeBin)
    params_cones,params_rods = model_params_orig(timeBin)
    params_cones['timeStep'] = 1e-3*(frameTime/upSampFac)
    params_rods['timeStep'] = 1e-3*(frameTime/upSampFac)
    

elif pr_mdl_name == 'clark':
    params_cones,params_rods = model_params_clark_init()
    params_cones['timeStep'] = frameTime/upSampFac
    params_rods['timeStep'] = frameTime/upSampFac



if pr_type == 'cones':
    params = params_cones
    y_lim = (-120,-40)
elif pr_type == 'rods':
    params = params_rods
    y_lim = (-10,2)

params['timeBin'] = timeBin

path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/',expDate,'datasets/'+folder)
fname_dataset = expDate+'_dataset_train_val_test_'+lightLevel+'.h5'
fname_data_train_val_test = os.path.join(path_dataset,fname_dataset)

path_dataset_save = os.path.join(path_dataset)#,'filterTest')

# if pr_mdl_name == 'rieke':
#     dataset_name = lightLevel+'-'+str(meanIntensity)+'_mdl-'+pr_mdl_name+'_s-'+str(params['sigma'])+'_p-'+str(params['phi'])+'_e-'+str(params['eta'])+'_k-'+str(params['k'])+'_h-'+str(params['h'])+'_b-'+str(params['beta'])+'_hc-'+str(params['hillcoef'])+'_gd-'+str(params['gdark'])+'_preproc-'+pr_type+'_norm-'+str(NORM)+'_tb-'+str(timeBin)+'_'+ode_solver+'_RF-'+str(ROLLING_FAC)
# elif pr_mdl_name == 'clark':
#     dataset_name = lightLevel+'-'+str(meanIntensity)+'_mdl-'+pr_mdl_name+'_a-'+str(params['alpha'])+pr_mdl_name+'_b-'+str(params['beta'])+'_g-'+str(params['gamma'])+'_y-'+str(params['tau_y'])+'_z-'+str(params['tau_z'])+'_r-'+str(params['tau_r'])+'_preproc-'+pr_type+'_norm-'+str(NORM)+'_rfac-'+str(ROLLING_FAC)+'_tb-'+str(timeBin)


# fname_dataset_save = expDate+'_dataset_train_val_test_'+dataset_name+'.h5'
# fname_dataset_save = os.path.join(path_dataset_save,fname_dataset_save)

data_train_orig,data_val_orig,data_test,data_quality,dataset_rr,parameters,resp_orig = load_h5Dataset(fname_data_train_val_test)

if DEBUG_MODE==1:
    nsamps_end = 10000  #10000
else:
    nsamps_end = data_train_orig.X.shape[0]-1 

frames_X_orig = data_train_orig.X[:nsamps_end]
X = frames_X_orig.reshape((frames_X_orig.shape[0]*frames_X_orig.shape[1]*frames_X_orig.shape[2]))
# Training data

stim_train,resp_train = run_model(pr_mdl_name,data_train_orig.X[:nsamps_end],data_train_orig.y[:nsamps_end],params,meanIntensity,upSampFac,downSampFac=downSampFac,n_discard=1000,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC,ode_solver=ode_solver,changeIntensities=False)
X_preproc = stim_train.reshape((stim_train.shape[0]*stim_train.shape[1]*stim_train.shape[2]))


if NORM==1:
    value_min = np.min(stim_train)
    value_max = np.max(stim_train)
    stim_train_norm = (stim_train - value_min)/(value_max-value_min)
    stim_train_med = np.nanmean(stim_train_norm)
    stim_train_norm = stim_train_norm - stim_train_med
else:
    stim_train_norm = stim_train
    
value_min = -0.05428597854503988  #tf.math.reduce_min(inputs)
value_max = 0.05213165772027483  #tf.math.reduce_max(inputs)
stim_train_med = 0.50609481460525 #tf.math.reduce_mean(R_norm)   


# %% OLD

def model_params_t2(timeBin = 1):
    
    ##  cones - trainable model
    params_cones = {}
    params_cones['sigma'] =  250 #22  # rhodopsin activity decay rate (1/sec) - default 22
    params_cones['phi'] =  40.7 #22    # phosphodiesterase activity decay rate (1/sec) - default 22
    params_cones['eta'] =  879  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
    params_cones['k'] =  0.01     # constant relating cGMP to current - default 0.02
    params_cones['h'] =  3       # cooperativity for cGMP->current - default 3
    params_cones['cdark'] =  1  # dark calcium concentration - default 1
    params_cones['beta'] = 110 # 9	  # rate constant for calcium removal in 1/sec - default 9
    params_cones['betaSlow'] =  10 #0	  
    params_cones['hillcoef'] =  2.64 #4  	  # cooperativity for cyclase, hill coef - default 4
    params_cones['hillaffinity'] =  1.51 #0.5   # hill affinity for cyclase - default 0.5
    params_cones['gamma'] =  10/timeBin #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_cones['timeStep'] =  1e-3  # freds default is 1e-4
    params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2
 
    # rods - mice - Retina 1 - trainable - path: /home/saad/data/analyses/data_kiersten/retina1/8ms_resamp/scotopic-1/PRFR_CNN2D_fixed/
    params_rods = {}
    params_rods['sigma'] = 8.35 #7.66  # rhodopsin activity decay rate (1/sec) 
    params_rods['phi'] =  8.44 #7.66     # phosphodiesterase activity decay rate (1/sec) 
    params_rods['eta'] = 28.8 #1.62	  # phosphodiesterase activation rate constant (1/sec) 
    params_rods['gdark'] = 20 # concentration of cGMP in darkness - default 20.5
    params_rods['k'] =  0.01    # constant relating cGMP to current - default 0.02
    params_rods['h'] =  3  # cooperativity for cGMP->current - default 3
    params_rods['cdark'] =  1  # dark calcium concentration - default 1
    params_rods['beta'] =  12.22 #25	  # rate constant for calcium removal in 1/sec - default 9
    params_rods['betaSlow'] =  0	  
    params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    params_rods['hillaffinity'] = 0.05 #0.22		# affinity for Ca2+
    params_rods['gamma'] =  10/timeBin #2.44/timeBin #2.44/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2


    return params_cones,params_rods



def model_params_t1(timeBin = 1):
    
    ##  cones - trainable model
    params_cones = {}
    params_cones['sigma'] =  250 #22  # rhodopsin activity decay rate (1/sec) - default 22
    params_cones['phi'] =  40.7 #22    # phosphodiesterase activity decay rate (1/sec) - default 22
    params_cones['eta'] =  879  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
    params_cones['k'] =  0.01     # constant relating cGMP to current - default 0.02
    params_cones['h'] =  3       # cooperativity for cGMP->current - default 3
    params_cones['cdark'] =  1  # dark calcium concentration - default 1
    params_cones['beta'] = 110 # 9	  # rate constant for calcium removal in 1/sec - default 9
    params_cones['betaSlow'] =  10 #0	  
    params_cones['hillcoef'] =  2.64 #4  	  # cooperativity for cyclase, hill coef - default 4
    params_cones['hillaffinity'] =  1.51 #0.5   # hill affinity for cyclase - default 0.5
    params_cones['gamma'] =  10/timeBin #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_cones['timeStep'] =  1e-3  # freds default is 1e-4
    params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2

 
    # rods - mice - Retina 1 - trainable
    params_rods = {}
    params_rods['sigma'] = 10.8 #7.66  # rhodopsin activity decay rate (1/sec) 
    params_rods['phi'] =  11.8 #7.66     # phosphodiesterase activity decay rate (1/sec) 
    params_rods['eta'] = 35.7 #1.62	  # phosphodiesterase activation rate constant (1/sec) 
    params_rods['gdark'] = 28 #28 # concentration of cGMP in darkness - default 20.5
    params_rods['k'] =  0.01    # constant relating cGMP to current - default 0.02
    params_rods['h'] =  3  # cooperativity for cGMP->current - default 3
    params_rods['cdark'] =  1  # dark calcium concentration - default 1
    params_rods['beta'] =  18.9 #25	  # rate constant for calcium removal in 1/sec - default 9
    params_rods['betaSlow'] =  0	  
    params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    params_rods['hillaffinity'] = 0.087 #0.22		# affinity for Ca2+
    params_rods['gamma'] =  2.44/timeBin #2.44/timeBin #2.44/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2


    # # rods - mice - Retina 2 (trainable)
    # params_rods = {}
    # params_rods['sigma'] = 14.44 # 7.66  # rhodopsin activity decay rate (1/sec) - default 22
    # params_rods['phi'] =  14.58  #7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_rods['eta'] = 20.44  #1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    # params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    # params_rods['h'] =  3 #3       # cooperativity for cGMP->current - default 3
    # params_rods['cdark'] =  1  # dark calcium concentration - default 1
    # params_rods['beta'] =  23 #25	  # rate constant for calcium removal in 1/sec - default 9
    # params_rods['betaSlow'] =  0	  
    # params_rods['hillcoef'] =  5.99  	  # cooperativity for cyclase, hill coef - default 4
    # params_rods['hillaffinity'] =  0.33		# affinity for Ca2+
    # params_rods['gamma'] =  5.4/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    # params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2


    return params_cones,params_rods