#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:05:08 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""


import jax
from jax import numpy as jnp
from flax import linen as nn


def model_definitions():
    """
        How to arrange the datasets depends on which model is being used
    """
    
    models_2D = ('CNN2D','CNN2D_MAXPOOL')
    
    models_3D = ('CNN_3D','PR_CNN3D')
    
    return (models_2D,models_3D)


def model_summary(mdl,inp_shape,console_kwargs={'width':180}):
    from flax.linen import summary

    inputs = jnp.ones([1]+list(inp_shape))    
    tabulate_fn = nn.tabulate(mdl, jax.random.PRNGKey(0),console_kwargs=console_kwargs)
    print(tabulate_fn(inputs,training=False))


class CNN2D(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    dtype : type
    # def __init__(self, **kwargs):
    #     self.__dict__.update(kwargs)

    @nn.compact
    def __call__(self,inputs,training: bool,**kwargs):       
        sigma=0.1       # sigma for noise
        y = jnp.moveaxis(inputs,1,-1)
        y = nn.LayerNorm(feature_axes=-1,reduction_axes=-1,epsilon=1e-7)(y)        # z-score the input across temporal dimension
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='VALID')(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

        if self.BatchNorm == 1:
            y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)

        y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
        y = nn.relu(y)
        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='VALID')(y)
            if self.BatchNorm == 1:
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
            y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)

        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='VALID')(y)
            if self.BatchNorm == 1:
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
            y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)
            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='VALID')(y)
            if self.BatchNorm == 1:
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
            y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)


                        
        y = y.reshape(y.shape[0],-1)
        y = nn.Dense(features=self.nout)(y)
        outputs = nn.softplus(y)
        
        return outputs

    

class CNN2D_MAXPOOL(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    dtype : type
    # def __init__(self, **kwargs):
    #     self.__dict__.update(kwargs)

    @nn.compact
    def __call__(self,inputs,training: bool,**kwargs):       
        sigma=0.01       # sigma for noise
        y = jnp.moveaxis(inputs,1,-1)       # Because jax is channels last
        # y = nn.LayerNorm(feature_axes=-1,reduction_axes=-1,epsilon=1e-7)(y)        # z-score the input across temporal dimension
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='VALID')(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

        if self.BatchNorm == 1:
            rgb = y.shape[1:]
            y = y.reshape(y.shape[0],-1)
            y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
            y = y.reshape(y.shape[0],*rgb)

            # y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)

        # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
        y = nn.relu(y)
        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='VALID')(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
                y = y.reshape(y.shape[0],*rgb)

            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)

        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='VALID')(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
                y = y.reshape(y.shape[0],*rgb)

            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)
            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='VALID')(y)
           
            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
                y = y.reshape(y.shape[0],*rgb)
                
            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)


                        
        y = y.reshape(y.shape[0],-1)
        y = nn.Dense(features=self.nout)(y)
        self.sow('intermediates', 'dense_activations', y)
        outputs = nn.softplus(y)
        
        return outputs

