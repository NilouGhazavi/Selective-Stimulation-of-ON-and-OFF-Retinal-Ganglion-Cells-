#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:42:39 2021

@author: saad
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model,regularizers
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, Activation, Flatten, Reshape,MaxPool3D, MaxPool2D, Permute, BatchNormalization, GaussianNoise,LayerNormalization
from tensorflow.keras.regularizers import l1, l2
import numpy as np
import math
from io import StringIO
import sys


# ----- HELPER FUNCS---- #
def model_definitions():
    """
        How to arrange the datasets depends on which model is being used
    """
    
    models_2D = ('CNN_2D','CNN_2D_NORM','CNN_2D_RAT','CNN_2D_TYPE1',
                 'PRFR_CNN2D','PR_CNN2D','PRFR_CNN2D_RODS','PRFR_CNN2D_RODSTRGAMMA',
                 'PRFR_CNN2D_RAT','PRFR_CNN2D_RAT_NHWC',
                 'BP_CNN2D','BP_CNN2D_PRFRTRAINABLEGAMMA',
                 'BP_CNN2D_MULTIBP','BP_CNN2D_MULTIBP_PRFRTRAINABLEGAMMA','BP_CNN2D_MULTIBP_PRFRTRAINABLEGAMMA_RODS')
    
    models_3D = ('CNN_3D','PR_CNN3D')
    
    return (models_2D,models_3D)

def get_model_memory_usage(batch_size, model):
    
    """
    Gets how much GPU memory will be required by the model.
    But doesn't work so good
    """
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem
    
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    
    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0
    
    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    
    return gbytes

def modelFileName(U=0,P=0,T=0,CB_n=0,C1_n=0,C1_s=0,C1_3d=0,C2_n=0,C2_s=0,C2_3d=0,C3_n=0,C3_s=0,C3_3d=0,C4_n=0,C4_s=0,C4_3d=0,BN=0,MP=0,LR=0,TR=0,with_TR=True,TRSAMPS=0,with_TRSAMPS=True):
    
    """
    Takes in data and model parameters, and parses them to 
    make the foldername where the model will be saved
    U : unit quality threshold
    P : Temporal dimension for the photoreceptor layer
    T : Temporal dimension for the first CNN
    C1, C2, C3 : CNN layers
    C_n : Number of channels in that CNN layer
    C_s : Convolution filter size (widht = height)
    C_3d : Size of the filter's third dimension in case CNNs are 3D Conv
    BN : BatchNormalization layer after each CNN (1=ON, 0=OFF)
    MP : MaxPool layer after first CNN (1=ON, 0=OFF)
    LR : Initial learning rate
    TR : Trial Number
    with_TR : Return filename with or without TR
    """
    
    def parse_param(key,val,fname):
        fname = fname+key+'-'+val+'_'    
        return fname

    fname = ''
    dict_params = {}
    
    if U>0.9:
        U = '%d'%U
    else:
        U = '%0.2f'%U
    fname = parse_param('U',U,fname)    
    
    if P>0:
        P = '%03d' %P
        fname = parse_param('P',P,fname)    
    
    T = '%03d' %T
    fname = parse_param('T',T,fname)    
    
    if CB_n>0:
        CB = '%02d'%CB_n
        dict_params['chans_bp'] = CB_n
        key = 'CB'
        fname = fname+key+'-'+eval(key)+'_'
    
    if C1_3d>0:
        C1 = '%02d-%02d-%02d'%(C1_n,C1_s,C1_3d)
        dict_params['filt1_3rdDim'] = C1_3d
    else:
        C1 = '%02d-%02d'%(C1_n,C1_s)
        
    dict_params['chan1_n'] = C1_n
    dict_params['filt1_size'] = C1_s
    key = 'C1'
    fname = fname+key+'-'+eval(key)+'_'    

    if C2_n>0:
        if C2_3d>0:
            C2 = '%02d-%02d-%02d'%(C2_n,C2_s,C2_3d)
            dict_params['filt2_3rdDim'] = C2_3d
        else:
            C2 = '%02d-%02d'%(C2_n,C2_s)
        
        key = 'C2'
        fname = fname+key+'-'+eval(key)+'_'    
    dict_params['chan2_n'] = C2_n
    dict_params['filt2_size'] = C2_s

        
    if C2_n>0 and C3_n>0:
        if C3_3d>0:
            C3 = '%02d-%02d-%02d'%(C3_n,C3_s,C3_3d)
            dict_params['filt3_3rdDim'] = C3_3d
        else:
            C3 = '%02d-%02d'%(C3_n,C3_s)
            
        key = 'C3'
        fname = fname+key+'-'+eval(key)+'_'    
    dict_params['chan3_n'] = C3_n
    dict_params['filt3_size'] = C3_s
    
    if C2_n>0 and C3_n>0 and C4_n>0:
        if C4_3d>0:
            C4 = '%02d-%02d-%02d'%(C4_n,C4_s,C4_3d)
            dict_params['filt4_3rdDim'] = C3_3d
        else:
            C4 = '%02d-%02d'%(C4_n,C4_s)
            
        key = 'C4'
        fname = fname+key+'-'+eval(key)+'_'    
    dict_params['chan4_n'] = C4_n
    dict_params['filt4_size'] = C4_s

            
    BN = '%d'%BN
    fname = parse_param('BN',BN,fname)    
    dict_params['BatchNorm'] = int(BN)

    MP = '%d'%MP
    fname = parse_param('MP',MP,fname)    
    dict_params['MaxPool'] = int(MP)
    
    # if LR>0:
    # LR = '%0.4f'%LR
    LR = str(LR)
    fname = parse_param('LR',LR,fname)    
        
        
    if with_TRSAMPS==True:
        TRSAMPS = '%03d'%TRSAMPS
        fname = parse_param('TRSAMPS',TRSAMPS,fname)    

    if with_TR==True:
        TR = '%02d'%TR
        fname = parse_param('TR',TR,fname)    
        
    fname_model = fname[:-1]
    
    
    return fname_model,dict_params

def get_layerFullNameStr(layer):
    tmp = sys.stdout
    layer_name = StringIO()
    sys.stdout = layer_name
    print(layer)
    sys.stdout = tmp
    layer_name = layer_name.getvalue()
    return layer_name

# %% Standard models
def cnn_2d(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])

    
    sigma = 0.1
    filt_temporal_width=inputs.shape[1]

    # first layer  
    if BatchNorm is True:
        y = inputs
        n1 = int(inputs.shape[-1])
        n2 = int(inputs.shape[-2])
        y = Reshape((filt_temporal_width, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    else:
        y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(inputs)
    
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        if BatchNorm is True: 
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  

        y = Activation('relu')(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        
        if BatchNorm is True: 
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))

        if y.shape[-1]<filt3_size:
            filt3_size = (filt3_size,y.shape[-1])
        elif y.shape[-2]<filt3_size:
            filt3_size = (y.shape[-2],filt3_size)
        else:
            filt3_size = filt3_size

        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)       
        y = Activation('relu')(GaussianNoise(sigma)(y))
        
        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'CNN_2D'
    return Model(inputs, outputs, name=mdl_name)

def cnn_2d_norm(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.1
    filt_temporal_width=inputs.shape[1]

    # first layer  
    y = inputs
    # y = LayerNormalization(axis=[1,2,3],epsilon=1e-7)(y)        # z-score the input
    y = LayerNormalization(epsilon=1e-7)(y)        # z-score the input
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        if y.shape[-1]<filt3_size:
            filt3_size = (filt3_size,y.shape[-1])
        elif y.shape[-2]<filt3_size:
            filt3_size = (y.shape[-2],filt3_size)
        else:
            filt3_size = filt3_size
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))

        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Fourth layer
    if mdl_params['chan4_n']>0:
        if y.shape[-1]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (mdl_params['filt4_size'],y.shape[-1])
        elif y.shape[-2]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (y.shape[-2],mdl_params['filt4_size'])
        else:
            mdl_params['filt4_size'] = mdl_params['filt4_size']
            
        y = Conv2D(mdl_params['chan4_n'], mdl_params['filt4_size'], data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))

        y = Activation('relu')(GaussianNoise(sigma)(y))

        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    # outputs = Activation('relu')(y)

    mdl_name = 'CNN_2D_NORM'
    return Model(inputs, outputs, name=mdl_name)

def cnn_2d_rat(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    # MaxPool = bool(kwargs['MaxPool'])
    MaxPool = kwargs['MaxPool']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.3
    filt_temporal_width=inputs.shape[1]

    # first layer  
    y = inputs
    # y = LayerNormalization(axis=[1,2,3],epsilon=1e-7)(y)        # z-score the input
    y = LayerNormalization(epsilon=1e-7)(y)        # z-score the input
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        if y.shape[-1]<filt3_size:
            filt3_size = (filt3_size,y.shape[-1])
        elif y.shape[-2]<filt3_size:
            filt3_size = (y.shape[-2],filt3_size)
        else:
            filt3_size = filt3_size
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))

        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Fourth layer
    if mdl_params['chan4_n']>0:
        if y.shape[-1]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (mdl_params['filt4_size'],y.shape[-1])
        elif y.shape[-2]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (y.shape[-2],mdl_params['filt4_size'])
        else:
            mdl_params['filt4_size'] = mdl_params['filt4_size']
            
        y = Conv2D(mdl_params['chan4_n'], mdl_params['filt4_size'], data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))

        y = Activation('relu')(GaussianNoise(sigma)(y))

        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    # y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-5))(y)  # original
    y = Dense(n_out,kernel_initializer='normal',kernel_regularizer=l2(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'CNN_2D_RAT'
    return Model(inputs, outputs, name=mdl_name)

def cnn_2d_type1(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    # MaxPool = bool(kwargs['MaxPool'])
    MaxPool = kwargs['MaxPool']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.3
    filt_temporal_width=inputs.shape[1]

    # first layer  
    y = inputs
    # y = LayerNormalization(axis=[1,2,3],epsilon=1e-7)(y)        # z-score the input
    y = LayerNormalization(epsilon=1e-7)(y)        # z-score the input
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-1))(y)
    
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)
    y = Activation('softplus')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-1))(y)                  
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Activation('softplus')(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        if y.shape[-1]<filt3_size:
            filt3_size = (filt3_size,y.shape[-1])
        elif y.shape[-2]<filt3_size:
            filt3_size = (y.shape[-2],filt3_size)
        else:
            filt3_size = filt3_size
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-1))(y)    
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))

        y = Activation('softplus')(GaussianNoise(sigma)(y))
    
    # Fourth layer
    if mdl_params['chan4_n']>0:
        if y.shape[-1]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (mdl_params['filt4_size'],y.shape[-1])
        elif y.shape[-2]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (y.shape[-2],mdl_params['filt4_size'])
        else:
            mdl_params['filt4_size'] = mdl_params['filt4_size']
            
        y = Conv2D(mdl_params['chan4_n'], mdl_params['filt4_size'], data_format="channels_first", kernel_regularizer=l2(1e-1))(y)    
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))

        y = Activation('softplus')(GaussianNoise(sigma)(y))

        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    # y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)  # original
    y = Dense(n_out,kernel_initializer='normal',kernel_regularizer=l2(1e-1))(y)
    outputs = Activation('sigmoid')(y)

    mdl_name = 'CNN_2D_TYPE1'
    return Model(inputs, outputs, name=mdl_name)



def cnn_3d(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, filt1_3rdDim=1, chan2_n=25, filt2_size=13, filt2_3rdDim=1, chan3_n=25, filt3_size=13, filt3_3rdDim=1, BatchNorm=True,MaxPool=True):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    filt1_3rdDim = kwargs['filt1_3rdDim']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    filt2_3rdDim = kwargs['filt2_3rdDim']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    filt3_3rdDim = kwargs['filt3_3rdDim']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])

    
    sigma = 0.1
    filt_temporal_width=inputs.shape[-1]

    # first layer  
    n1 = int(inputs.shape[-2])
    n2 = int(inputs.shape[-3])
    y = Reshape((inputs.shape[1],n2, n1,filt_temporal_width))(BatchNormalization(axis=-1)(Flatten()(inputs)))
    y = Conv3D(chan1_n, (filt1_size,filt1_size,filt1_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    if MaxPool:
        y = MaxPool3D(2,data_format='channels_first')(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        n1 = int(y.shape[-2])
        n2 = int(y.shape[-3])
        y = Reshape((y.shape[1],y.shape[2], y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Conv3D(chan2_n, (filt2_size,filt2_size,filt2_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        # y = MaxPool3D(2,data_format='channels_first')(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))
       
    # Third layer
    if chan3_n>0:
        y = Reshape((y.shape[1],y.shape[2], y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Conv3D(chan3_n, (filt3_size,filt3_size,filt3_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        # y = MaxPool3D(2,data_format='channels_first')(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Dense layer
    y = Flatten()(y)
    y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    
    mdl_name = 'CNN_3D'
    return Model(inputs, outputs, name=mdl_name)
 

# %% CLASS: NORMALIZAZE
class Normalize(tf.keras.layers.Layer):
    """
    BatchNorm is where you calculate normalization factors for each dimension seperately based on
    the batch data
    LayerNorm is where you calculate the normalization factors based on channels and dimensions
    Normalize combines both: you calculate normalization factors based on channels, dimensions and the batch
    """
    def __init__(self,units=1):
        super(Normalize,self).__init__()
        self.units = units
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
         })
         return config   
             
    def call(self,inputs):
        value_min = tf.math.reduce_min(inputs)
        value_max = tf.math.reduce_max(inputs)
        R_norm = (inputs - value_min)/(value_max-value_min)
        R_mean = tf.math.reduce_mean(R_norm)       
        R_norm = R_norm - R_mean
        return R_norm

class Normalize_PRDA(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(Normalize_PRDA,self).__init__()
        self.units = units
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
         })
         return config   
             
    def call(self,inputs):
        value_min = 0.004992888155901156 #tf.math.reduce_min(inputs)
        value_max = 0.02672805318583508 #tf.math.reduce_max(inputs)
        R_norm = (inputs - value_min)/(value_max-value_min)
        R_mean = 0.5233899120505345 #tf.math.reduce_mean(R_norm)       
        R_norm = R_norm - R_mean
        return R_norm

class Normalize_PRDA_GF(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(Normalize_PRDA_GF,self).__init__()
        self.units = units
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
         })
         return config   
             
    def call(self,inputs):
        # without normalized cone responses
        # value_min = 1.080396013135178
        # value_max = 1.1141200609927957
        # R_mean = 0.40979439338018137
        # --- with normalized cone responses
        value_min = -0.05428597854503988  #tf.math.reduce_min(inputs)
        value_max = 0.05213165772027483  #tf.math.reduce_max(inputs)
        R_mean = 0.50609481460525 #tf.math.reduce_mean(R_norm)   
        
        R_norm = (inputs - value_min)/(value_max-value_min)
        R_norm = R_norm - R_mean
        return R_norm

class Normalize_PRFR_SI(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(Normalize_PRFR_SI,self).__init__()
        self.units = units
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
         })
         return config   
             
    def call(self,inputs):
        # value_min = -103 #tf.math.reduce_min(inputs)
        # value_max = -87 #tf.math.reduce_max(inputs)
        # R_norm = (inputs - value_min)/(value_max-value_min)
        # R_mean = 0.54 #tf.math.reduce_mean(R_norm)       
        
        value_min = -111 
        value_max = -105
        R_norm = (inputs - value_min)/(value_max-value_min)
        R_mean = 0.51

        R_norm = R_norm - R_mean
        return R_norm

class Normalize_PRFR_GF(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(Normalize_PRFR_GF,self).__init__()
        self.units = units
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
         })
         return config   
             
    def call(self,inputs):
        value_min = -113.211676833235 #tf.math.reduce_min(inputs)
        value_max = -81.1005869186533 #tf.math.reduce_max(inputs)
        R_norm = (inputs - value_min)/(value_max-value_min)
        R_mean =  0.43907212331830137 #tf.math.reduce_mean(R_norm)       
        R_norm = R_norm - R_mean
        return R_norm
    
class Normalize_MEAN(tf.keras.layers.Layer):
    def __init__(self,mean_val=0):
        super(Normalize_MEAN,self).__init__()
        self.mean_val = mean_val
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "mean_val": self.mean_val,
         })
         return config   
             
    def call(self,inputs):
        R_mean =  self.mean_val #tf.math.reduce_mean(R_norm)       
        if R_mean==0:
            R_mean = tf.math.reduce_mean(inputs)       
        R_norm = inputs - R_mean
        return R_norm


class Normalize_multichan(tf.keras.layers.Layer):
    """
    BatchNorm is where you calculate normalization factors for each dimension seperately based on
    the batch data
    LayerNorm is where you calculate the normalization factors based on channels and dimensions
    Normalize_multichan calculates normalization factors based on all dimensions for each channel seperately
    """
    
    def __init__(self,units=1):
        super(Normalize_multichan,self).__init__()
        self.units = units
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
         })
         return config   
             
    def call(self,inputs):
        inputs_perChan = tf.reshape(inputs,(-1,inputs.shape[-1]))
        value_min = tf.reduce_min(inputs_perChan,axis=0)
        value_max = tf.reduce_max(inputs_perChan,axis=0)
        
        # value_min = tf.expand_dims(value_min,axis=0)
        R_norm = (inputs - value_min[None,None,None,None,:])/(value_max[None,None,None,None,:]-value_min[None,None,None,None,:])
        R_norm_perChan = tf.reshape(R_norm,(-1,R_norm.shape[-1]))
        R_mean = tf.reduce_mean(R_norm_perChan,axis=0)       
        R_norm = R_norm - R_mean[None,None,None,None,:]
        return R_norm

# %% Clarks PR Model

def generate_simple_filter(tau,n,t):
    f = (t**n)*tf.math.exp(-t/tau) # functional form in paper
    f = (f/tau**(n+1))/tf.math.exp(tf.math.lgamma(n+1)) # normalize appropriately
    # print(t.shape)
    # print(n.shape)
    # print(tau.shape)

    return f

def conv_oper(x,kernel_1D):
    spatial_dims = x.shape[-1]
    x_reshaped = tf.expand_dims(x,axis=2)
    kernel_1D = tf.squeeze(kernel_1D)
    kernel_1D = tf.reverse(kernel_1D,[0])
    tile_fac = tf.constant([spatial_dims])
    kernel_reshaped = tf.tile(kernel_1D,tile_fac)
    kernel_reshaped = tf.reshape(kernel_reshaped,(1,spatial_dims,1,kernel_1D.shape[-1]))
    kernel_reshaped = tf.experimental.numpy.moveaxis(kernel_reshaped,-1,0)
    pad_vec = [[0,0],[kernel_1D.shape[-1]-1,0],[0,0],[0,0]]
    conv_output = tf.nn.depthwise_conv2d(x_reshaped,kernel_reshaped,strides=[1,1,1,1],padding=pad_vec,data_format='NHWC')
    return conv_output

class photoreceptor_DA(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(photoreceptor_DA,self).__init__()
        self.units = units
            
    def build(self,input_shape):
        alpha_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(16.2) #tf.keras.initializers.Constant(1.) #tf.random_normal_initializer(mean=1)
        self.alpha = tf.Variable(name='alpha',initial_value=alpha_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        beta_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(-13.46) #tf.keras.initializers.Constant(0.36) #tf.random_normal_initializer(mean=0.36)
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        gamma_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(16.49) #tf.keras.initializers.Constant(0.448) #tf.random_normal_initializer(mean=0.448)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        # self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        tauY_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(0.928) #tf.keras.initializers.Constant(10.) #tf.random_normal_initializer(mean=2) #tf.random_uniform_initializer(minval=1)
        self.tauY = tf.Variable(name='tauY',initial_value=tauY_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        tauZ_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(0.008) #tf.keras.initializers.Constant(166) #tf.random_normal_initializer(mean=166) #tf.random_uniform_initializer(minval=100)
        self.tauZ = tf.Variable(name='tauZ',initial_value=tauZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        nY_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(1.439) #tf.keras.initializers.Constant(4.33) #tf.random_normal_initializer(mean=4.33) #tf.random_uniform_initializer(minval=1)
        self.nY = tf.Variable(name='nY',initial_value=nY_init(shape=(1,self.units),dtype='float32'),trainable=True)   
        
        nZ_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(0.29) #tf.keras.initializers.Constant(1) #tf.random_uniform_initializer(minval=1)
        self.nZ = tf.Variable(name='nZ',initial_value=nZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        
        tauY_mulFac = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(10.) 
        self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        tauZ_mulFac = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(10.) 
        self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
        nY_mulFac = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(10.) 
        self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
        nZ_mulFac = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(10.) 
        self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
    def call(self,inputs):
       
        timeBin = 8
        
        alpha =  self.alpha / timeBin
        beta = self.beta / timeBin
        # beta = tf.sigmoid(float(self.beta / timeBin))
        gamma =  self.gamma
        # gamma =  tf.sigmoid(float(self.gamma))
        tau_y =  (self.tauY_mulFac*self.tauY) / timeBin
        tau_z =  (self.tauZ_mulFac*self.tauZ) / timeBin
        n_y =  (self.nY_mulFac*self.nY)
        n_z =  (self.nZ_mulFac*self.nZ)
        
        t = tf.range(0,1000/timeBin,dtype='float32')
        
        Ky = generate_simple_filter(tau_y,n_y,t)   
        Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))
       
        y_tf = conv_oper(inputs,Ky)
        z_tf = conv_oper(inputs,Kz)

        outputs = (alpha*y_tf)/(1+(beta*z_tf))
        
        return outputs

# KERAS MODEL
def pr_cnn2d(inputs,n_out,**kwargs): #(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
        
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])

    sigma = 0.1
    
    keras_prLayer = photoreceptor_DA(units=1)
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = keras_prLayer(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    
    y = Normalize_PRDA(units=1)(y)
    
    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        # y = BatchNormalization(axis=1)(y)   
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'PR_CNN2D'
    return Model(inputs, outputs, name=mdl_name)

# %% Fred's PR Model

@tf.function(autograph=True,experimental_relax_shapes=True)
def riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark):
    darkCurrent = gdark**cgmphill * cgmp2cur/2
    gdark = (2 * darkCurrent / cgmp2cur) **(1/cgmphill)
    
    cur2ca = beta * cdark / darkCurrent                # get q using steady state
    smax = eta/phi * gdark * (1 + (cdark / hillaffinity) **hillcoef)		# get smax using steady state
    
    n_spatialDims = X_fun.shape[-1]
    tme = tf.range(0,X_fun.shape[1],dtype='float32')*TimeStep
    NumPts = tme.shape[0]
    
# initial conditions   
    g_prev = gdark+(X_fun[:,0,:]*0)
    s_prev = (gdark * eta/phi)+(X_fun[:,0,:]*0)
    c_prev = cdark+(X_fun[:,0,:]*0)
    cslow_prev = cdark+(X_fun[:,0,:]*0)
    r_prev = X_fun[:,0,:] * gamma / sigma
    p_prev = (eta + r_prev)/phi

    g = tf.TensorArray(tf.float32,size=NumPts)
    g.write(0,X_fun[:,0,:]*0)
    
    # solve difference equations
    for pnt in tf.range(1,NumPts):
        r_curr = r_prev + TimeStep * (-1 * sigma * r_prev)
        r_curr = r_curr + gamma * X_fun[:,pnt-1,:]
        p_curr = p_prev + TimeStep * (r_prev + eta - phi * p_prev)
        # c_curr = c_prev + TimeStep * (cur2ca * (cgmp2cur * g_prev **cgmphill)/2 - beta * c_prev)
        c_curr = c_prev + TimeStep * (cur2ca * cgmp2cur * g_prev**cgmphill /(1+(cslow_prev/cdark)) - beta * c_prev)
        cslow_curr = cslow_prev - TimeStep * (betaSlow * (cslow_prev-c_prev))
        s_curr = smax / (1 + (c_curr / hillaffinity) **hillcoef)
        g_curr = g_prev + TimeStep * (s_prev - p_prev * g_prev)

        g = g.write(pnt,g_curr)
        
        
        # update prev values to current
        g_prev = g_curr#[0,:]
        s_prev = s_curr#[0,:]
        c_prev = c_curr#[0,:]
        p_prev = p_curr
        r_prev = r_curr
        cslow_prev = cslow_curr#[0,:]
    
    g = g.stack()
    g = tf.transpose(g,(1,0,2))
    outputs = -(cgmp2cur * g **cgmphill)/2
    
    return outputs
 
class photoreceptor_REIKE(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(photoreceptor_REIKE,self).__init__()
        self.units = units
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self,input_shape):
        sigma_init = tf.keras.initializers.Constant(1.) # 22
        self.sigma = tf.Variable(name='sigma',initial_value=sigma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        sigma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.sigma_scaleFac = tf.Variable(name='sigma_scaleFac',initial_value=sigma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        phi_init = tf.keras.initializers.Constant(1.) #22
        self.phi = tf.Variable(name='phi',initial_value=phi_init(shape=(1,self.units),dtype='float32'),trainable=True)
        phi_scaleFac = tf.keras.initializers.Constant(10.) 
        self.phi_scaleFac = tf.Variable(name='phi_scaleFac',initial_value=phi_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
       
        eta_init = tf.keras.initializers.Constant(1.) #2000
        self.eta = tf.Variable(name='eta',initial_value=eta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        eta_scaleFac = tf.keras.initializers.Constant(1000.) 
        self.eta_scaleFac = tf.Variable(name='eta_scaleFac',initial_value=eta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        beta_init = tf.keras.initializers.Constant(1.) #9
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        beta_scaleFac = tf.keras.initializers.Constant(10.) 
        self.beta_scaleFac = tf.Variable(name='beta_scaleFac',initial_value=beta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        cgmp2cur_init = tf.keras.initializers.Constant(0.01) # 0.01
        self.cgmp2cur = tf.Variable(name='cgmp2cur',initial_value=cgmp2cur_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cgmphill_init = tf.keras.initializers.Constant(3.)  # 3
        self.cgmphill = tf.Variable(name='cgmphill',initial_value=cgmphill_init(shape=(1,self.units),dtype='float32'),trainable=False)
        cgmphill_scaleFac = tf.keras.initializers.Constant(1.) 
        self.cgmphill_scaleFac = tf.Variable(name='cgmphill_scaleFac',initial_value=cgmphill_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cdark_init = tf.keras.initializers.Constant(1.)
        self.cdark = tf.Variable(name='cdark',initial_value=cdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        betaSlow_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(1.) # 0
        self.betaSlow = tf.Variable(name='betaSlow',initial_value=betaSlow_init(shape=(1,self.units),dtype='float32'),trainable=True)
        betaSlow_scaleFac = tf.keras.initializers.Constant(1.) 
        self.betaSlow_scaleFac = tf.Variable(name='betaSlow_scaleFac',initial_value=betaSlow_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillcoef_init = tf.keras.initializers.Constant(4.) #tf.keras.initializers.Constant(1.) # 4
        self.hillcoef = tf.Variable(name='hillcoef',initial_value=hillcoef_init(shape=(1,self.units),dtype='float32'),trainable=True)
        hillcoef_scaleFac = tf.keras.initializers.Constant(1.) 
        self.hillcoef_scaleFac = tf.Variable(name='hillcoef_scaleFac',initial_value=hillcoef_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillaffinity_init = tf.keras.initializers.Constant(1.) # 0.5
        self.hillaffinity = tf.Variable(name='hillaffinity',initial_value=hillaffinity_init(shape=(1,self.units),dtype='float32'),trainable=True)
        hillaffinity_scaleFac = tf.keras.initializers.Constant(1.) 
        self.hillaffinity_scaleFac = tf.Variable(name='hillaffinity_scaleFac',initial_value=hillaffinity_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        gamma_init = tf.keras.initializers.Constant(1.)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=False)
        gamma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        gdark_init = tf.keras.initializers.Constant(0.28)    # 28 for cones; 20 for rods 
        self.gdark = tf.Variable(name='gdark',initial_value=gdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        self.timeBin = 8 # find a way to fix this in the model  #tf.Variable(name='timeBin',initial_value=timeBin(shape=(1,self.units),dtype='float32'),trainable=False)

 
    def call(self,inputs):
        X_fun = inputs

        timeBin = float(self.timeBin) # ms
        frameTime = 8 # ms
        upSamp_fac = int(frameTime/timeBin)
        TimeStep = 1e-3*timeBin
        
        if upSamp_fac>1:
            X_fun = tf.keras.backend.repeat_elements(X_fun,upSamp_fac,axis=1) 
            X_fun = X_fun/upSamp_fac     # appropriate scaling for photons/ms

        sigma = self.sigma * self.sigma_scaleFac
        phi = self.phi * self.phi_scaleFac
        eta = self.eta * self.eta_scaleFac
        cgmp2cur = self.cgmp2cur
        cgmphill = self.cgmphill * self.cgmphill_scaleFac
        cdark = self.cdark
        beta = self.beta * self.beta_scaleFac
        betaSlow = self.betaSlow * self.betaSlow_scaleFac
        hillcoef = self.hillcoef * self.hillcoef_scaleFac
        hillaffinity = self.hillaffinity * self.hillaffinity_scaleFac
        gamma = (self.gamma*self.gamma_scaleFac)/timeBin
        gdark = self.gdark*100
        
        
        outputs = riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark)
        
        if upSamp_fac>1:
            outputs = outputs[:,upSamp_fac-1::upSamp_fac]
            
        return outputs

# KERAS MODEL
def prfr_cnn2d(inputs,n_out,**kwargs): #(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):

    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])
    
    sigma = 0.1
    
    # keras_prLayer = photoreceptor_REIKE(units=1)
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_REIKE(units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y = Normalize_PRFR_SI(units=1)(y)
    # y = Normalize(units=1)(y)
    
    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        # y = BatchNormalization(axis=1)(y)   
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
            
        # if MaxPool is True:
        #     y = MaxPool2D(2,data_format='channels_first')(y)

        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRFR_CNN2D'
    return Model(inputs, outputs, name=mdl_name)

# %% Fred's PR RODS Model

class photoreceptor_RODS_REIKE(tf.keras.layers.Layer):
    def __init__(self,units=1,kernel_regularizer=None):
        super(photoreceptor_RODS_REIKE,self).__init__()
        self.units = units
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self,input_shape):
        sigma_init = tf.keras.initializers.Constant(0.707) #1
        self.sigma = tf.Variable(name='sigma',initial_value=sigma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        sigma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.sigma_scaleFac = tf.Variable(name='sigma_scaleFac',initial_value=sigma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        phi_init = tf.keras.initializers.Constant(0.707) #1
        self.phi = tf.Variable(name='phi',initial_value=phi_init(shape=(1,self.units),dtype='float32'),trainable=True)
        phi_scaleFac = tf.keras.initializers.Constant(10.) 
        self.phi_scaleFac = tf.Variable(name='phi_scaleFac',initial_value=phi_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
       
        eta_init = tf.keras.initializers.Constant(0.0253) #0.2
        self.eta = tf.Variable(name='eta',initial_value=eta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        eta_scaleFac = tf.keras.initializers.Constant(100.) 
        self.eta_scaleFac = tf.Variable(name='eta_scaleFac',initial_value=eta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        beta_init = tf.keras.initializers.Constant(0.25) #0.15
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        beta_scaleFac = tf.keras.initializers.Constant(100.) 
        self.beta_scaleFac = tf.Variable(name='beta_scaleFac',initial_value=beta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        cgmp2cur_init = tf.keras.initializers.Constant(0.01) # 0.01
        self.cgmp2cur = tf.Variable(name='cgmp2cur',initial_value=cgmp2cur_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cgmphill_init = tf.keras.initializers.Constant(0.3)  # 3
        self.cgmphill = tf.Variable(name='cgmphill',initial_value=cgmphill_init(shape=(1,self.units),dtype='float32'),trainable=False)
        cgmphill_scaleFac = tf.keras.initializers.Constant(10.) 
        self.cgmphill_scaleFac = tf.Variable(name='cgmphill_scaleFac',initial_value=cgmphill_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cdark_init = tf.keras.initializers.Constant(1.)
        self.cdark = tf.Variable(name='cdark',initial_value=cdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        betaSlow_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(1.) # 0
        self.betaSlow = tf.Variable(name='betaSlow',initial_value=betaSlow_init(shape=(1,self.units),dtype='float32'),trainable=False)
        betaSlow_scaleFac = tf.keras.initializers.Constant(1.) 
        self.betaSlow_scaleFac = tf.Variable(name='betaSlow_scaleFac',initial_value=betaSlow_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillcoef_init = tf.keras.initializers.Constant(0.4) #0.52 #tf.keras.initializers.Constant(1.) # 4
        self.hillcoef = tf.Variable(name='hillcoef',initial_value=hillcoef_init(shape=(1,self.units),dtype='float32'),trainable=False)
        hillcoef_scaleFac = tf.keras.initializers.Constant(10.) 
        self.hillcoef_scaleFac = tf.Variable(name='hillcoef_scaleFac',initial_value=hillcoef_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillaffinity_init = tf.keras.initializers.Constant(0.5) # 0.26
        self.hillaffinity = tf.Variable(name='hillaffinity',initial_value=hillaffinity_init(shape=(1,self.units),dtype='float32'),trainable=False)
        hillaffinity_scaleFac = tf.keras.initializers.Constant(1.) 
        self.hillaffinity_scaleFac = tf.Variable(name='hillaffinity_scaleFac',initial_value=hillaffinity_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        # gamma_range = (0.01,1)
        # gamma_init = tf.keras.initializers.Constant(0.1)
        # self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True,constraint=lambda x: tf.clip_by_value(x,gamma_range[0],gamma_range[1]))
        # gamma_scaleFac = tf.keras.initializers.Constant(1000.) 
        # self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        gamma_init = tf.keras.initializers.Constant(1)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=False)
        gamma_scaleFac = tf.keras.initializers.Constant(100.) 
        self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        gdark_init = tf.keras.initializers.Constant(0.15)    # 20 for rods 
        self.gdark = tf.Variable(name='gdark',initial_value=gdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        gdark_scaleFac = tf.keras.initializers.Constant(100)    # 28 for cones; 20 for rods 
        self.gdark_scaleFac = tf.Variable(name='gdark_scaleFac',initial_value=gdark_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        self.timeBin = 8 # find a way to fix this in the model  #tf.Variable(name='timeBin',initial_value=timeBin(shape=(1,self.units),dtype='float32'),trainable=False)

 
    def call(self,inputs):
        X_fun = inputs

        timeBin = float(self.timeBin) # ms
        frameTime = 8 # ms
        upSamp_fac = int(frameTime/timeBin)
        TimeStep = 1e-3*timeBin
        
        if upSamp_fac>1:
            X_fun = tf.keras.backend.repeat_elements(X_fun,upSamp_fac,axis=1) 
            X_fun = X_fun/upSamp_fac     # appropriate scaling for photons/ms

        sigma = self.sigma * self.sigma_scaleFac
        phi = self.phi * self.phi_scaleFac
        eta = self.eta * self.eta_scaleFac
        cgmp2cur = self.cgmp2cur
        cgmphill = self.cgmphill * self.cgmphill_scaleFac
        cdark = self.cdark
        beta = self.beta * self.beta_scaleFac
        betaSlow = self.betaSlow * self.betaSlow_scaleFac
        hillcoef = self.hillcoef * self.hillcoef_scaleFac
        hillaffinity = self.hillaffinity * self.hillaffinity_scaleFac
        gamma = (self.gamma*self.gamma_scaleFac)/timeBin
        gdark = self.gdark*self.gdark_scaleFac
        
        outputs = riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark)
        
        if upSamp_fac>1:
            outputs = outputs[:,upSamp_fac-1::upSamp_fac]
            
        return outputs

# KERAS MODEL
def prfr_cnn2d_rods(inputs,n_out,**kwargs): #(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):

    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])
    
    sigma = 0.1
    
    # keras_prLayer = photoreceptor_REIKE(units=1)
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_RODS_REIKE(units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    
    # y = Normalize_PRFR_MEAN(units=1)(y)
    # y = Normalize_PRFR_GF(units=1)(y)
    # y = Normalize(units=1)(y)
    y = LayerNormalization(axis=[1,2,3],epsilon=1e-7)(y)
    # y = LayerNormalization()(y)
    
    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        # y = BatchNormalization(axis=1)(y)   
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
            
        # if MaxPool is True:
        #     y = MaxPool2D(2,data_format='channels_first')(y)

        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRFR_CNN2D_RODS'
    return Model(inputs, outputs, name=mdl_name)


# %% Bipolar MULTICHAN


def generate_simple_filter_multichan(tau,n,t):

    t_shape = t.shape[0]
    t = tf.tile(t,tf.constant([tau.shape[-1]], tf.int32))
    t = tf.reshape(t,(tau.shape[-1],t_shape))
    t = tf.transpose(t)
    f = (t**n[:,None])*tf.math.exp(-t/tau[:,None]) # functional form in paper
    rgb = tau**(n+1)
    f = (f/rgb[:,None])/tf.math.exp(tf.math.lgamma(n+1))[:,None] # normalize appropriately
    # print(t.shape)
    # print(n.shape)
    # print(tau.shape)
   
    return f

""" test filter 

    tau = tf.constant([[1]],dtype=tf.float32)
    n = tf.constant([[1]],dtype=tf.float32)
    t = tf.range(0,1000/timeBin,dtype='float32')
    t_shape = t.shape[0]
    t = tf.tile(t,tf.constant([tau.shape[-1]], tf.int32))
    t = tf.reshape(t,(tau.shape[-1],t_shape))
    t = tf.transpose(t)
    tN = np.squeeze(t.eval(session=tf.compat.v1.Session()))
    a = t**n; aN = np.squeeze(a.eval(session=tf.compat.v1.Session()))
    f = (t**n)*tf.math.exp(-t/tau) # functional form in paper
    rgb = tau**(n+1)
    f = (f/rgb)/tf.math.exp(tf.math.lgamma(n+1)) # normalize appropriately
    
    f = np.squeeze(f.eval(session=tf.compat.v1.Session()))
    plt.plot(f)


"""

def conv_oper_multichan(x,kernel_1D):
    spatial_dims = x.shape[-1]
    x_reshaped = tf.expand_dims(x,axis=2)
    kernel_1D = tf.squeeze(kernel_1D,axis=0)
    kernel_1D = tf.reverse(kernel_1D,[0])
    tile_fac = tf.constant([spatial_dims,1])
    kernel_reshaped = tf.tile(kernel_1D,(tile_fac))
    kernel_reshaped = tf.reshape(kernel_reshaped,(1,spatial_dims,kernel_1D.shape[0],kernel_1D.shape[-1]))
    kernel_reshaped = tf.experimental.numpy.moveaxis(kernel_reshaped,-2,0)
    pad_vec = [[0,0],[kernel_1D.shape[0]-1,0],[0,0],[0,0]]
    conv_output = tf.nn.depthwise_conv2d(x_reshaped,kernel_reshaped,strides=[1,1,1,1],padding=pad_vec,data_format='NHWC')
    # conv_output = tf.reshape(conv_output,(conv_output.shape[0],conv_output.shape[1],conv_output.shape[2],spatial_dims,kernel_1D.shape[-1]))
    return conv_output


class photoreceptor_DA_multichan(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(photoreceptor_DA_multichan,self).__init__()
        self.units = units
            
    def build(self,input_shape):
        alpha_init = tf.keras.initializers.Constant(0.5) #tf.keras.initializers.Constant(16.2) #tf.keras.initializers.Constant(1.) #tf.random_normal_initializer(mean=1)
        self.alpha = tf.Variable(name='alpha',initial_value=alpha_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        beta_init = tf.keras.initializers.Constant(0.5) #tf.keras.initializers.Constant(-13.46) #tf.keras.initializers.Constant(0.36) #tf.random_normal_initializer(mean=0.36)
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        gamma_init = tf.keras.initializers.Constant(0.5) #tf.keras.initializers.Constant(16.49) #tf.keras.initializers.Constant(0.448) #tf.random_normal_initializer(mean=0.448)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        # self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        
        tauY_init = tf.keras.initializers.Constant(.3) #tf.keras.initializers.Constant(0.928) #tf.keras.initializers.Constant(10.) #tf.random_normal_initializer(mean=2) #tf.random_uniform_initializer(minval=1)
        self.tauY = tf.Variable(name='tauY',initial_value=tauY_init(shape=(1,self.units),dtype='float32'),trainable=True)
        tauY_mulFac = tf.keras.initializers.Constant(100.) #tf.keras.initializers.Constant(10.) 
        self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)

        tauZ_init = tf.keras.initializers.Constant(.8) #tf.keras.initializers.Constant(0.008) #tf.keras.initializers.Constant(166) #tf.random_normal_initializer(mean=166) #tf.random_uniform_initializer(minval=100)
        self.tauZ = tf.Variable(name='tauZ',initial_value=tauZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
        tauZ_mulFac = tf.keras.initializers.Constant(100.) #tf.keras.initializers.Constant(10.) 
        self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        
        nY_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(1.439) #tf.keras.initializers.Constant(4.33) #tf.random_normal_initializer(mean=4.33) #tf.random_uniform_initializer(minval=1)
        self.nY = tf.Variable(name='nY',initial_value=nY_init(shape=(1,self.units),dtype='float32'),trainable=True)   
        nY_mulFac = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(10.) 
        self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        nZ_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(0.29) #tf.keras.initializers.Constant(1) #tf.random_uniform_initializer(minval=1)
        self.nZ = tf.Variable(name='nZ',initial_value=nZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
        nZ_mulFac = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(10.) 
        self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
    
    def call(self,inputs):
       
        timeBin = 8
        
        alpha =  self.alpha / timeBin
        beta = self.beta / timeBin
        # beta = tf.sigmoid(float(self.beta / timeBin))
        gamma =  self.gamma
        # gamma =  tf.sigmoid(float(self.gamma))
        tau_y =  (self.tauY_mulFac*self.tauY) / timeBin
        tau_z =  (self.tauZ_mulFac*self.tauZ) / timeBin
        n_y =  (self.nY_mulFac*self.nY)
        n_z =  (self.nZ_mulFac*self.nZ)
        
        t = tf.range(0,1000/timeBin,dtype='float32')
        
        Ky = generate_simple_filter_multichan(tau_y,n_y,t)   
        Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter_multichan(tau_z,n_z,t))
        
        y_tf = conv_oper_multichan(inputs,Ky)
        z_tf = conv_oper_multichan(inputs,Kz)
                
        y_tf_reshape = tf.reshape(y_tf,(-1,y_tf.shape[1],y_tf.shape[2],inputs.shape[-1],alpha.shape[-1]))
        z_tf_reshape = tf.reshape(z_tf,(-1,z_tf.shape[1],z_tf.shape[2],inputs.shape[-1],alpha.shape[-1]))
    
        outputs = (alpha[None,None,:,None,:]*y_tf_reshape)/(1+(beta[None,None,:,None,:]*z_tf_reshape))
        
        return outputs



def bp_cnn2d_multibp(inputs,n_out,**kwargs): # BP --> 3D CNN --> 2D CNN
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chans_bp = kwargs['chans_bp']
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])
    
    # chans_bp = chan1_n

    sigma = 0.1
    
    y = inputs
    
    # add layer normalization for photoreceptor inputs
    y = LayerNormalization()(y)
    
    
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(y)
    y = photoreceptor_DA_multichan(units=chans_bp)(y)
    y = y[:,:,0,:,:]
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1],chans_bp))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:,:]
    
    y = LayerNormalization()(y)
    y = Permute((4,2,3,1))(y)   # Channels first
    
    # CNN - first layer
    y = Conv3D(chan1_n, (filt1_size,filt1_size,filt_temporal_width), data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    y = tf.keras.backend.squeeze(y,-1) # y[:,:,:,:,0]
    if BatchNorm is True:
        y = LayerNormalization()(y)
    #     n1 = int(y.shape[-1])
    #     n2 = int(y.shape[-2])
    #     y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan3_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            y = LayerNormalization()(y)
        #     n1 = int(y.shape[-1])
        #     n2 = int(y.shape[-2])
        #     y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            y = LayerNormalization()(y)
        #     n1 = int(y.shape[-1])
        #     n2 = int(y.shape[-2])
        #     y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = LayerNormalization()(y)
        # y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'BP_CNN2D_MULTIBP'
    return Model(inputs, outputs, name=mdl_name)

# %% Bipolar SINGLE CHAN


class bipolar(tf.keras.layers.Layer):
    def __init__(self,units=1,kernel_regularizer=None):
        super(bipolar,self).__init__()
        self.units = units
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
            
    def build(self,input_shape):
        
        alpha_range = (0.01,1)
        alpha_init = tf.keras.initializers.Constant(0.05) 
        self.alpha = self.add_weight(name='alpha',initializer=alpha_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,alpha_range[0],alpha_range[1]))
        alpha_mulFac = tf.keras.initializers.Constant(10.) 
        self.alpha_mulFac = self.add_weight(name='alpha_mulFac',initializer=alpha_mulFac,shape=[1,self.units],trainable=False)
        
        beta_range = (0.00,0.1)
        beta_init = tf.keras.initializers.Constant(0.05)# 
        self.beta = self.add_weight(name='beta',initializer=beta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,beta_range[0],beta_range[1]))
        beta_mulFac = tf.keras.initializers.Constant(10.) 
        self.beta_mulFac = self.add_weight(name='beta_mulFac',initializer=beta_mulFac,shape=[1,self.units],trainable=False)
        
        gamma_range = (0.01,0.1)
        gamma_init = tf.keras.initializers.Constant(0.05)# 
        self.gamma = self.add_weight(name='gamma',initializer=gamma_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,gamma_range[0],gamma_range[1]))
        gamma_mulFac = tf.keras.initializers.Constant(10.) 
        self.gamma_mulFac = self.add_weight(name='gamma_mulFac',initializer=gamma_mulFac,shape=[1,self.units],trainable=False)
        
        kappa_range = (0.00,0.01)
        kappa_init = tf.keras.initializers.RandomUniform(minval=kappa_range[0],maxval=kappa_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.kappa = self.add_weight(name='kappa',initializer=kappa_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,kappa_range[0],kappa_range[1]))
        kappa_mulFac = tf.keras.initializers.Constant(1000.) 
        self.kappa_mulFac = self.add_weight(name='kappa_mulFac',initializer=kappa_mulFac,shape=[1,self.units],trainable=False)

        tauY_range = (0.01,1.)
        tauY_init = tf.keras.initializers.Constant(0.3)# 
        self.tauY = self.add_weight(name='tauY',initializer=tauY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauY_range[0],tauY_range[1]))
        tauY_mulFac = tf.keras.initializers.Constant(100.) #tf.keras.initializers.Constant(10.) 
        self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
 
        nY_range = (1e-5,0.5)
        nY_init = tf.keras.initializers.Constant(0.1)# 
        self.nY = self.add_weight(name='nY',initializer=nY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nY_range[0],nY_range[1]))
        nY_mulFac = tf.keras.initializers.Constant(10.) 
        self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)

        tauZ_range = (0.01,10.)
        tauZ_init = tf.keras.initializers.Constant(0.8)# 0.5
        self.tauZ = self.add_weight(name='tauZ',initializer=tauZ_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauZ_range[0],tauZ_range[1]))
        tauZ_mulFac = tf.keras.initializers.Constant(100.) 
        self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        nZ_range = (1e-5,1.)
        nZ_init = tf.keras.initializers.Constant(0.1) #tf.keras.initializers.RandomUniform(minval=nZ_range[0],maxval=nZ_range[1])  #tf.keras.initializers.Constant(0.01)# 
        self.nZ = self.add_weight(name='nZ',initializer=nZ_init,shape=[1,self.units],trainable=False,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nZ_range[0],nZ_range[1]))
        nZ_mulFac = tf.keras.initializers.Constant(10.) 
        self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        # alpha_init = tf.keras.initializers.Constant(0.5) #tf.keras.initializers.Constant(16.2) #tf.keras.initializers.Constant(1.) #tf.random_normal_initializer(mean=1)
        # self.alpha = tf.Variable(name='alpha',initial_value=alpha_init(shape=(1,self.units),dtype='float32'),trainable=True)

        # beta_init = tf.keras.initializers.Constant(0.5) #tf.keras.initializers.Constant(-13.46) #tf.keras.initializers.Constant(0.36) #tf.random_normal_initializer(mean=0.36)
        # self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)

        # gamma_init = tf.keras.initializers.Constant(0.5) #tf.keras.initializers.Constant(16.49) #tf.keras.initializers.Constant(0.448) #tf.random_normal_initializer(mean=0.448)
        # self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)

        
        # tauY_init = tf.keras.initializers.Constant(.3) #tf.keras.initializers.Constant(0.928) #tf.keras.initializers.Constant(10.) #tf.random_normal_initializer(mean=2) #tf.random_uniform_initializer(minval=1)
        # self.tauY = tf.Variable(name='tauY',initial_value=tauY_init(shape=(1,self.units),dtype='float32'),trainable=True)
        # tauY_mulFac = tf.keras.initializers.Constant(100.) #tf.keras.initializers.Constant(10.) 
        # self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)

        # tauZ_init = tf.keras.initializers.Constant(.8) #tf.keras.initializers.Constant(0.008) #tf.keras.initializers.Constant(166) #tf.random_normal_initializer(mean=166) #tf.random_uniform_initializer(minval=100)
        # self.tauZ = tf.Variable(name='tauZ',initial_value=tauZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
        # tauZ_mulFac = tf.keras.initializers.Constant(100.) #tf.keras.initializers.Constant(10.) 
        # self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        
        # nY_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(1.439) #tf.keras.initializers.Constant(4.33) #tf.random_normal_initializer(mean=4.33) #tf.random_uniform_initializer(minval=1)
        # self.nY = tf.Variable(name='nY',initial_value=nY_init(shape=(1,self.units),dtype='float32'),trainable=True)   
        # nY_mulFac = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(10.) 
        # self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        # nZ_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(0.29) #tf.keras.initializers.Constant(1) #tf.random_uniform_initializer(minval=1)
        # self.nZ = tf.Variable(name='nZ',initial_value=nZ_init(shape=(1,self.units),dtype='float32'),trainable=True)
        # nZ_mulFac = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(10.) 
        # self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
    def call(self,inputs):
       
        timeBin = 8
        
        alpha =  self.alpha*self.alpha_mulFac / timeBin
        beta = self.beta*self.beta_mulFac / timeBin
        gamma =  self.gamma*self.gamma_mulFac
        kappa = self.kappa*self.kappa_mulFac

        tau_y =  (self.tauY_mulFac*self.tauY) / timeBin
        tau_z =  (self.tauZ_mulFac*self.tauZ) / timeBin
        n_y =  (self.nY_mulFac*self.nY)
        n_z =  (self.nZ_mulFac*self.nZ)
        
        t = tf.range(0,1000/timeBin,dtype='float32')
        
        Ky = generate_simple_filter(tau_y,n_y,t)   
        Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))
       
        y_tf = conv_oper(inputs,Ky)
        z_tf = conv_oper(inputs,Kz)

        # outputs = (alpha*y_tf)/(1+(beta*z_tf))
        outputs = (alpha*y_tf)/(kappa+1e-6+(beta*z_tf))
        
        return outputs

def bp_cnn2d(inputs,n_out,**kwargs): # BP --> 2D CNN --> 2D CNN
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chans_bp = 1
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])
    
    # chans_bp = chan1_n

    sigma = 0.1
    
    y = inputs
    
    # add layer normalization for photoreceptor inputs
    y = LayerNormalization()(y)
    
    
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(y)
    y = bipolar(units=1)(y)
    y = y[:,:,0,:]
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    
    y = LayerNormalization()(y)
    # y = BatchNormalization()(y)
    # y = Permute((4,2,3,1))(y)   # Channels first
    
    # CNN - first layer
    y = Conv2D(chan1_n,filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    # y = tf.keras.backend.squeeze(y,-1) # y[:,:,:,:,0]
    if BatchNorm is True:
        # y = LayerNormalization()(y)
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan3_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            # y = LayerNormalization()(y)
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            # y = LayerNormalization()(y)
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        # y = LayerNormalization()(y)
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'BP_CNN2D'
    return Model(inputs, outputs, name=mdl_name)

# %% PR-RODS-G + SignleChan BP

class photoreceptor_REIKE_RODS_fixed(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(photoreceptor_REIKE_RODS_fixed,self).__init__()
        self.units = units
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self,input_shape):
        sigma_init = tf.keras.initializers.Constant(0.707) #1
        self.sigma = tf.Variable(name='sigma',initial_value=sigma_init(shape=(1,self.units),dtype='float32'),trainable=False)
        sigma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.sigma_scaleFac = tf.Variable(name='sigma_scaleFac',initial_value=sigma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        phi_init = tf.keras.initializers.Constant(0.707) #1
        self.phi = tf.Variable(name='phi',initial_value=phi_init(shape=(1,self.units),dtype='float32'),trainable=False)
        phi_scaleFac = tf.keras.initializers.Constant(10.) 
        self.phi_scaleFac = tf.Variable(name='phi_scaleFac',initial_value=phi_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
       
        eta_init = tf.keras.initializers.Constant(0.0253) #0.2
        self.eta = tf.Variable(name='eta',initial_value=eta_init(shape=(1,self.units),dtype='float32'),trainable=False)
        eta_scaleFac = tf.keras.initializers.Constant(100.) 
        self.eta_scaleFac = tf.Variable(name='eta_scaleFac',initial_value=eta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        beta_init = tf.keras.initializers.Constant(0.25) #0.15
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=False)
        beta_scaleFac = tf.keras.initializers.Constant(100.) 
        self.beta_scaleFac = tf.Variable(name='beta_scaleFac',initial_value=beta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        cgmp2cur_init = tf.keras.initializers.Constant(0.01) # 0.01
        self.cgmp2cur = tf.Variable(name='cgmp2cur',initial_value=cgmp2cur_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cgmphill_init = tf.keras.initializers.Constant(0.3)  # 3
        self.cgmphill = tf.Variable(name='cgmphill',initial_value=cgmphill_init(shape=(1,self.units),dtype='float32'),trainable=False)
        cgmphill_scaleFac = tf.keras.initializers.Constant(10.) 
        self.cgmphill_scaleFac = tf.Variable(name='cgmphill_scaleFac',initial_value=cgmphill_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cdark_init = tf.keras.initializers.Constant(1.)
        self.cdark = tf.Variable(name='cdark',initial_value=cdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        betaSlow_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(1.) # 0
        self.betaSlow = tf.Variable(name='betaSlow',initial_value=betaSlow_init(shape=(1,self.units),dtype='float32'),trainable=False)
        betaSlow_scaleFac = tf.keras.initializers.Constant(1.) 
        self.betaSlow_scaleFac = tf.Variable(name='betaSlow_scaleFac',initial_value=betaSlow_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillcoef_init = tf.keras.initializers.Constant(0.4) #0.52 #tf.keras.initializers.Constant(1.) # 4
        self.hillcoef = tf.Variable(name='hillcoef',initial_value=hillcoef_init(shape=(1,self.units),dtype='float32'),trainable=False)
        hillcoef_scaleFac = tf.keras.initializers.Constant(10.) 
        self.hillcoef_scaleFac = tf.Variable(name='hillcoef_scaleFac',initial_value=hillcoef_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillaffinity_init = tf.keras.initializers.Constant(0.5) # 0.26
        self.hillaffinity = tf.Variable(name='hillaffinity',initial_value=hillaffinity_init(shape=(1,self.units),dtype='float32'),trainable=False)
        hillaffinity_scaleFac = tf.keras.initializers.Constant(1.) 
        self.hillaffinity_scaleFac = tf.Variable(name='hillaffinity_scaleFac',initial_value=hillaffinity_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        gamma_init = tf.keras.initializers.Constant(1)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        gamma_scaleFac = tf.keras.initializers.Constant(100.) 
        self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        gdark_init = tf.keras.initializers.Constant(0.15)    # 20 for rods 
        self.gdark = tf.Variable(name='gdark',initial_value=gdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        gdark_scaleFac = tf.keras.initializers.Constant(100)    # 28 for cones; 20 for rods 
        self.gdark_scaleFac = tf.Variable(name='gdark_scaleFac',initial_value=gdark_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        self.timeBin = 8 # find a way to fix this in the model  #tf.Variable(name='timeBin',initial_value=timeBin(shape=(1,self.units),dtype='float32'),trainable=False)

 
    def call(self,inputs):
        X_fun = inputs

        timeBin = float(self.timeBin) # ms
        frameTime = 8 # ms
        upSamp_fac = int(frameTime/timeBin)
        TimeStep = 1e-3*timeBin
        
        if upSamp_fac>1:
            X_fun = tf.keras.backend.repeat_elements(X_fun,upSamp_fac,axis=1) 
            X_fun = X_fun/upSamp_fac     # appropriate scaling for photons/ms

        sigma = self.sigma * self.sigma_scaleFac
        phi = self.phi * self.phi_scaleFac
        eta = self.eta * self.eta_scaleFac
        cgmp2cur = self.cgmp2cur
        cgmphill = self.cgmphill * self.cgmphill_scaleFac
        cdark = self.cdark
        beta = self.beta * self.beta_scaleFac
        betaSlow = self.betaSlow * self.betaSlow_scaleFac
        hillcoef = self.hillcoef * self.hillcoef_scaleFac
        hillaffinity = self.hillaffinity * self.hillaffinity_scaleFac
        gamma = (self.gamma*self.gamma_scaleFac)/timeBin
        gdark = self.gdark*self.gdark_scaleFac
        

        outputs = riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark)
        
        if upSamp_fac>1:
            outputs = outputs[:,upSamp_fac-1::upSamp_fac]
            
        return outputs

# single chan bp
def bp_cnn2d_prfrtrainablegamma(inputs,n_out,**kwargs):
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])
    
    sigma = 0.1
    
    # RIEKE PR Layer
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_REIKE_RODS_fixed(units=1)(y)
    y = LayerNormalization()(y)

    # Clark's layer for BP
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(y)
    y = bipolar(units=1)(y)
    y = y[:,:,0,:]
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    
    y = LayerNormalization()(y)
    # y = Permute((4,2,3,1))(y)   # Channels first
    
    # CNN - first layer
    y = Conv2D(chan1_n,filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    # y = tf.keras.backend.squeeze(y,-1) # y[:,:,:,:,0]
    if BatchNorm is True:
        # y = LayerNormalization()(y)
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan3_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            # y = LayerNormalization()(y)
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            # y = LayerNormalization()(y)
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        # y = LayerNormalization()(y)
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'BP_CNN2D_PRFRTRAINABLEGAMMA'
    return Model(inputs, outputs, name=mdl_name)




# %% PR (gamma) + Multichannel BP

class photoreceptor_REIKE_fixed(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(photoreceptor_REIKE_fixed,self).__init__()
        self.units = units
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self,input_shape):
        sigma_init = tf.keras.initializers.Constant(2.2) # 22
        self.sigma = tf.Variable(name='sigma',initial_value=sigma_init(shape=(1,self.units),dtype='float32'),trainable=False)
        sigma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.sigma_scaleFac = tf.Variable(name='sigma_scaleFac',initial_value=sigma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        phi_init = tf.keras.initializers.Constant(2.2) #22
        self.phi = tf.Variable(name='phi',initial_value=phi_init(shape=(1,self.units),dtype='float32'),trainable=False)
        phi_scaleFac = tf.keras.initializers.Constant(10.) 
        self.phi_scaleFac = tf.Variable(name='phi_scaleFac',initial_value=phi_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
       
        eta_init = tf.keras.initializers.Constant(2.) #2000
        self.eta = tf.Variable(name='eta',initial_value=eta_init(shape=(1,self.units),dtype='float32'),trainable=False)
        eta_scaleFac = tf.keras.initializers.Constant(1000.) 
        self.eta_scaleFac = tf.Variable(name='eta_scaleFac',initial_value=eta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        beta_init = tf.keras.initializers.Constant(0.9) #9
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=False)
        beta_scaleFac = tf.keras.initializers.Constant(10.) 
        self.beta_scaleFac = tf.Variable(name='beta_scaleFac',initial_value=beta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        cgmp2cur_init = tf.keras.initializers.Constant(0.01) # 0.01
        self.cgmp2cur = tf.Variable(name='cgmp2cur',initial_value=cgmp2cur_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cgmphill_init = tf.keras.initializers.Constant(3.)  # 3
        self.cgmphill = tf.Variable(name='cgmphill',initial_value=cgmphill_init(shape=(1,self.units),dtype='float32'),trainable=False)
        cgmphill_scaleFac = tf.keras.initializers.Constant(1.) 
        self.cgmphill_scaleFac = tf.Variable(name='cgmphill_scaleFac',initial_value=cgmphill_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cdark_init = tf.keras.initializers.Constant(1.)
        self.cdark = tf.Variable(name='cdark',initial_value=cdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        betaSlow_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(1.) # 0
        self.betaSlow = tf.Variable(name='betaSlow',initial_value=betaSlow_init(shape=(1,self.units),dtype='float32'),trainable=False)
        betaSlow_scaleFac = tf.keras.initializers.Constant(1.) 
        self.betaSlow_scaleFac = tf.Variable(name='betaSlow_scaleFac',initial_value=betaSlow_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillcoef_init = tf.keras.initializers.Constant(4.) #tf.keras.initializers.Constant(1.) # 4
        self.hillcoef = tf.Variable(name='hillcoef',initial_value=hillcoef_init(shape=(1,self.units),dtype='float32'),trainable=False)
        hillcoef_scaleFac = tf.keras.initializers.Constant(1.) 
        self.hillcoef_scaleFac = tf.Variable(name='hillcoef_scaleFac',initial_value=hillcoef_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillaffinity_init = tf.keras.initializers.Constant(0.5) # 0.5
        self.hillaffinity = tf.Variable(name='hillaffinity',initial_value=hillaffinity_init(shape=(1,self.units),dtype='float32'),trainable=False)
        hillaffinity_scaleFac = tf.keras.initializers.Constant(1.) 
        self.hillaffinity_scaleFac = tf.Variable(name='hillaffinity_scaleFac',initial_value=hillaffinity_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        gamma_init = tf.keras.initializers.Constant(1.)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        gamma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        gdark_init = tf.keras.initializers.Constant(0.28)    # 28 for cones; 20 for rods 
        self.gdark = tf.Variable(name='gdark',initial_value=gdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        self.timeBin = 8 # find a way to fix this in the model  #tf.Variable(name='timeBin',initial_value=timeBin(shape=(1,self.units),dtype='float32'),trainable=False)

 
    def call(self,inputs):
        X_fun = inputs

        timeBin = float(self.timeBin) # ms
        frameTime = 8 # ms
        upSamp_fac = int(frameTime/timeBin)
        TimeStep = 1e-3*timeBin
        
        if upSamp_fac>1:
            X_fun = tf.keras.backend.repeat_elements(X_fun,upSamp_fac,axis=1) 
            X_fun = X_fun/upSamp_fac     # appropriate scaling for photons/ms

        sigma = self.sigma * self.sigma_scaleFac
        phi = self.phi * self.phi_scaleFac
        eta = self.eta * self.eta_scaleFac
        cgmp2cur = self.cgmp2cur
        cgmphill = self.cgmphill * self.cgmphill_scaleFac
        cdark = self.cdark
        beta = self.beta * self.beta_scaleFac
        betaSlow = self.betaSlow * self.betaSlow_scaleFac
        hillcoef = self.hillcoef * self.hillcoef_scaleFac
        hillaffinity = self.hillaffinity * self.hillaffinity_scaleFac
        gamma = (self.gamma*self.gamma_scaleFac)/timeBin
        gdark = self.gdark*100
        

        outputs = riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark)
        
        if upSamp_fac>1:
            outputs = outputs[:,upSamp_fac-1::upSamp_fac]
            
        return outputs

def bp_cnn2d_multibp_prfrtrainablegamma(inputs,n_out,**kwargs):
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])
    
    sigma = 0.1
    
    # RIEKE PR Layer
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_REIKE_fixed(units=1)(y)
    y = Normalize(units=1)(y)

    # Clark's layer for BP
    y = photoreceptor_DA_multichan(units=chan1_n)(y)
    y = y[:,:,0,:,:]
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:,:]
    y = Normalize(units=1)(y)
    y = Permute((4,2,3,1))(y)   # Channels first
    
    # CNN - first layer
    y = Conv3D(chan2_n, (filt2_size,filt2_size,filt_temporal_width), data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    y = tf.keras.backend.squeeze(y,-1) # y[:,:,:,:,0]
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'BP_CNN2D_MULTIBP_PRFRTRAINABLEGAMMA'
    return Model(inputs, outputs, name=mdl_name)




# %% PR-BP-TrainableVersions RODs

class photoreceptor_REIKE_fixed_rods(tf.keras.layers.Layer):
    def __init__(self,units=1):
        super(photoreceptor_REIKE_fixed_rods,self).__init__()
        self.units = units
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config


    def build(self,input_shape):
        sigma_init = tf.keras.initializers.Constant(1.) # 22
        self.sigma = tf.Variable(name='sigma',initial_value=sigma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        sigma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.sigma_scaleFac = tf.Variable(name='sigma_scaleFac',initial_value=sigma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        phi_init = tf.keras.initializers.Constant(1.) #22
        self.phi = tf.Variable(name='phi',initial_value=phi_init(shape=(1,self.units),dtype='float32'),trainable=True)
        phi_scaleFac = tf.keras.initializers.Constant(10.) 
        self.phi_scaleFac = tf.Variable(name='phi_scaleFac',initial_value=phi_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
       
        eta_init = tf.keras.initializers.Constant(.02) #2000
        self.eta = tf.Variable(name='eta',initial_value=eta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        eta_scaleFac = tf.keras.initializers.Constant(1000.) 
        self.eta_scaleFac = tf.Variable(name='eta_scaleFac',initial_value=eta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        beta_init = tf.keras.initializers.Constant(1) #9
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        beta_scaleFac = tf.keras.initializers.Constant(10.) 
        self.beta_scaleFac = tf.Variable(name='beta_scaleFac',initial_value=beta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        cgmp2cur_init = tf.keras.initializers.Constant(0.01) # 0.01
        self.cgmp2cur = tf.Variable(name='cgmp2cur',initial_value=cgmp2cur_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cgmphill_init = tf.keras.initializers.Constant(3.)  # 3
        self.cgmphill = tf.Variable(name='cgmphill',initial_value=cgmphill_init(shape=(1,self.units),dtype='float32'),trainable=False)
        cgmphill_scaleFac = tf.keras.initializers.Constant(1.) 
        self.cgmphill_scaleFac = tf.Variable(name='cgmphill_scaleFac',initial_value=cgmphill_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cdark_init = tf.keras.initializers.Constant(1.)
        self.cdark = tf.Variable(name='cdark',initial_value=cdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        betaSlow_init = tf.keras.initializers.Constant(1.) #tf.keras.initializers.Constant(1.) # 0
        self.betaSlow = tf.Variable(name='betaSlow',initial_value=betaSlow_init(shape=(1,self.units),dtype='float32'),trainable=False)
        betaSlow_scaleFac = tf.keras.initializers.Constant(1.) 
        self.betaSlow_scaleFac = tf.Variable(name='betaSlow_scaleFac',initial_value=betaSlow_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillcoef_init = tf.keras.initializers.Constant(4.) #tf.keras.initializers.Constant(1.) # 4
        self.hillcoef = tf.Variable(name='hillcoef',initial_value=hillcoef_init(shape=(1,self.units),dtype='float32'),trainable=True)
        hillcoef_scaleFac = tf.keras.initializers.Constant(1.) 
        self.hillcoef_scaleFac = tf.Variable(name='hillcoef_scaleFac',initial_value=hillcoef_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillaffinity_init = tf.keras.initializers.Constant(0.4) # 0.5
        self.hillaffinity = tf.Variable(name='hillaffinity',initial_value=hillaffinity_init(shape=(1,self.units),dtype='float32'),trainable=True)
        hillaffinity_scaleFac = tf.keras.initializers.Constant(1.) 
        self.hillaffinity_scaleFac = tf.Variable(name='hillaffinity_scaleFac',initial_value=hillaffinity_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        gamma_init = tf.keras.initializers.Constant(1.)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        gamma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        gdark_init = tf.keras.initializers.Constant(0.28)    # 28 for cones; 20 for rods 
        self.gdark = tf.Variable(name='gdark',initial_value=gdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        self.timeBin = 8 # find a way to fix this in the model  #tf.Variable(name='timeBin',initial_value=timeBin(shape=(1,self.units),dtype='float32'),trainable=False)

 
    def call(self,inputs):
        X_fun = inputs

        timeBin = float(self.timeBin) # ms
        frameTime = 8 # ms
        upSamp_fac = int(frameTime/timeBin)
        TimeStep = 1e-3*timeBin
        
        if upSamp_fac>1:
            X_fun = tf.keras.backend.repeat_elements(X_fun,upSamp_fac,axis=1) 
            X_fun = X_fun/upSamp_fac     # appropriate scaling for photons/ms

        sigma = self.sigma * self.sigma_scaleFac
        phi = self.phi * self.phi_scaleFac
        eta = self.eta * self.eta_scaleFac
        cgmp2cur = self.cgmp2cur
        cgmphill = self.cgmphill * self.cgmphill_scaleFac
        cdark = self.cdark
        beta = self.beta * self.beta_scaleFac
        betaSlow = self.betaSlow * self.betaSlow_scaleFac
        hillcoef = self.hillcoef * self.hillcoef_scaleFac
        hillaffinity = self.hillaffinity * self.hillaffinity_scaleFac
        gamma = (self.gamma*self.gamma_scaleFac)/timeBin
        gdark = self.gdark*100
        

        outputs = riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark)
        
        if upSamp_fac>1:
            outputs = outputs[:,upSamp_fac-1::upSamp_fac]
            
        return outputs


def bp_cnn2d_multibp_prfrtrainablegamma_rods(inputs,n_out,**kwargs):
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])
    
    sigma = 0.1
    
    # RIEKE PR Layer
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_REIKE_fixed_rods(units=1)(y)
    y = Normalize(units=1)(y)

    # Clark's layer for BP
    y = photoreceptor_DA_multichan(units=chan1_n)(y)
    y = y[:,:,0,:,:]
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:,:]
    y = Normalize(units=1)(y)
    y = Permute((4,2,3,1))(y)   # Channels first
    
    # CNN - first layer
    y = Conv3D(chan2_n, (filt2_size,filt2_size,filt_temporal_width), data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    y = tf.keras.backend.squeeze(y,-1) # y[:,:,:,:,0]
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'BP_CNN2D_MULTIBP_PRFRTRAINABLEGAMMA_RODS'
    return Model(inputs, outputs, name=mdl_name)
 
    
 
# %% RAT-PR

class photoreceptor_RAT_REIKE(tf.keras.layers.Layer):
    def __init__(self,units=1,kernel_regularizer=None):
        super(photoreceptor_RAT_REIKE,self).__init__()
        self.units = units
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self,input_shape):
        sigma_init = tf.keras.initializers.Constant(1.124) #1
        self.sigma = tf.Variable(name='sigma',initial_value=sigma_init(shape=(1,self.units),dtype='float32'),trainable=False)
        sigma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.sigma_scaleFac = tf.Variable(name='sigma_scaleFac',initial_value=sigma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        phi_init = tf.keras.initializers.Constant(1.124) #1
        self.phi = tf.Variable(name='phi',initial_value=phi_init(shape=(1,self.units),dtype='float32'),trainable=False)
        phi_scaleFac = tf.keras.initializers.Constant(10.) 
        self.phi_scaleFac = tf.Variable(name='phi_scaleFac',initial_value=phi_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
       
        eta_init = tf.keras.initializers.Constant(5.76) #0.2
        self.eta = tf.Variable(name='eta',initial_value=eta_init(shape=(1,self.units),dtype='float32'),trainable=False)
        eta_scaleFac = tf.keras.initializers.Constant(100.) 
        self.eta_scaleFac = tf.Variable(name='eta_scaleFac',initial_value=eta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        beta_init = tf.keras.initializers.Constant(0.347) #10
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=False)
        beta_scaleFac = tf.keras.initializers.Constant(10.) 
        self.beta_scaleFac = tf.Variable(name='beta_scaleFac',initial_value=beta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        cgmp2cur_init = tf.keras.initializers.Constant(0.01) # 0.01
        self.cgmp2cur = tf.Variable(name='cgmp2cur',initial_value=cgmp2cur_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cgmphill_init = tf.keras.initializers.Constant(0.3)  # 3
        self.cgmphill = tf.Variable(name='cgmphill',initial_value=cgmphill_init(shape=(1,self.units),dtype='float32'),trainable=False)
        cgmphill_scaleFac = tf.keras.initializers.Constant(10.) 
        self.cgmphill_scaleFac = tf.Variable(name='cgmphill_scaleFac',initial_value=cgmphill_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cdark_init = tf.keras.initializers.Constant(1.)
        self.cdark = tf.Variable(name='cdark',initial_value=cdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        betaSlow_init = tf.keras.initializers.Constant(0.04) #tf.keras.initializers.Constant(1.) # 0
        self.betaSlow = tf.Variable(name='betaSlow',initial_value=betaSlow_init(shape=(1,self.units),dtype='float32'),trainable=False)
        betaSlow_scaleFac = tf.keras.initializers.Constant(1.) 
        self.betaSlow_scaleFac = tf.Variable(name='betaSlow_scaleFac',initial_value=betaSlow_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillcoef_init = tf.keras.initializers.Constant(0.4) #0.52 #tf.keras.initializers.Constant(1.) # 4
        self.hillcoef = tf.Variable(name='hillcoef',initial_value=hillcoef_init(shape=(1,self.units),dtype='float32'),trainable=False)
        hillcoef_scaleFac = tf.keras.initializers.Constant(10.) 
        self.hillcoef_scaleFac = tf.Variable(name='hillcoef_scaleFac',initial_value=hillcoef_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillaffinity_init = tf.keras.initializers.Constant(0.03326) # 0.26
        self.hillaffinity = tf.Variable(name='hillaffinity',initial_value=hillaffinity_init(shape=(1,self.units),dtype='float32'),trainable=False)
        hillaffinity_scaleFac = tf.keras.initializers.Constant(10.) 
        self.hillaffinity_scaleFac = tf.Variable(name='hillaffinity_scaleFac',initial_value=hillaffinity_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        # gamma_range = (0.01,1)
        # gamma_init = tf.keras.initializers.Constant(0.1)
        # self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True,constraint=lambda x: tf.clip_by_value(x,gamma_range[0],gamma_range[1]))
        # gamma_scaleFac = tf.keras.initializers.Constant(1000.) 
        # self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        gamma_init = tf.keras.initializers.Constant(1)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        gamma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        gdark_init = tf.keras.initializers.Constant(0.20)    # 20 for rods 
        self.gdark = tf.Variable(name='gdark',initial_value=gdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        gdark_scaleFac = tf.keras.initializers.Constant(100)    # 28 for cones; 20 for rods 
        self.gdark_scaleFac = tf.Variable(name='gdark_scaleFac',initial_value=gdark_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        # self.timeBin = tf.Variable(name='timeBin',initial_value=tf.constant(16,dtype='int32'),trainable=False)
        self.timeBin = 16 # find a way to fix this in the model  #tf.Variable(name='timeBin',initial_value=timeBin(shape=(1,self.units),dtype='float32'),trainable=False)

 
    def call(self,inputs):
        X_fun = inputs

        timeBin = float(self.timeBin) # ms
        frameTime = 16 # ms
        upSamp_fac = int(frameTime/timeBin)
        TimeStep = 1e-3*timeBin
        
        if upSamp_fac>1:
            X_fun = tf.keras.backend.repeat_elements(X_fun,upSamp_fac,axis=1) 
            X_fun = X_fun/upSamp_fac     # appropriate scaling for photons/ms

        sigma = self.sigma * self.sigma_scaleFac
        phi = self.phi * self.phi_scaleFac
        eta = self.eta * self.eta_scaleFac
        cgmp2cur = self.cgmp2cur
        cgmphill = self.cgmphill * self.cgmphill_scaleFac
        cdark = self.cdark
        beta = self.beta * self.beta_scaleFac
        betaSlow = self.betaSlow * self.betaSlow_scaleFac
        hillcoef = self.hillcoef * self.hillcoef_scaleFac
        hillaffinity = self.hillaffinity * self.hillaffinity_scaleFac
        gamma = (self.gamma*self.gamma_scaleFac)/timeBin
        gdark = self.gdark*self.gdark_scaleFac
        
        outputs = riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark)
        
        if upSamp_fac>1:
            outputs = outputs[:,upSamp_fac-1::upSamp_fac]
            
        return outputs

# KERAS MODEL
def prfr_cnn2d_rat(inputs,n_out,**kwargs): #(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):

    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    if MaxPool==1:
        MaxPool=2
    
    sigma = 0.1
    
    # keras_prLayer = photoreceptor_REIKE(units=1)
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_RAT_REIKE(units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    # y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y = y[:,-filt_temporal_width:,:,:]

    
    y = LayerNormalization(epsilon=1e-7)(y)
    
    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        # y = BatchNormalization(axis=1)(y)   
        
    if MaxPool > 1:
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   

        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    # y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3))(y)
    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRFR_CNN2D_RAT'
    return Model(inputs, outputs, name=mdl_name)

def prfr_cnn2d_rat_nhwc(inputs,n_out,**kwargs): #(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):

    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    if MaxPool==1:
        MaxPool=2
    
    sigma = 0.1
    
    # keras_prLayer = photoreceptor_REIKE(units=1)
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_RAT_REIKE(units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    # y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y = y[:,-filt_temporal_width:,:,:]
    y = Permute((2,3,1))(y)  # channels last
    
    y = LayerNormalization(epsilon=1e-7)(y)
    
    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_last", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    if BatchNorm is True:
        rgb = y.shape[1:]
        y = Reshape((rgb))(BatchNormalization(axis=-1)(Flatten()(y)))
        # y = BatchNormalization(axis=1)(y)   
        
    if MaxPool > 1:
        y = MaxPool2D(MaxPool,data_format='channels_last')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_last", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            rgb = y.shape[1:]
            y = Reshape((rgb))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   

        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_last", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            rgb = y.shape[1:]
            y = Reshape((rgb))(BatchNormalization(axis=-1)(Flatten()(y)))
            # y = BatchNormalization(axis=1)(y)   
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    # y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3))(y)
    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRFR_CNN2D_RAT'
    return Model(inputs, outputs, name=mdl_name)