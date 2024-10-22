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
    
    models_2D = ('CNN2D','CNN2D_LNORMSPAT','CNN2D_BNORM','CNN2D_BNORMSPAT','CNN2D_NONORM','CNN2D_NEWBN','CNN2D_MAXPOOL',
                 'CNN2D_AF',
                 'PRFR_CNN2D','PRFR_CNN2D_NOLN','PRFR_CNN2D_ARU',
                 'PRFR_LN_CNN2D',
                 'PRDA_CNN2D')
    
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

# %%  Standard models
def cnn2d(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
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
    y = LayerNormalization(axis=-3,epsilon=1e-7,trainable=False)(y)        # z-score the input across temporal dimension
    # y = LayerNormalization(epsilon=1e-7)(y)        # z-score the input
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)
        
    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  
        
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
            
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
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)

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
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)

        y = Activation('relu')(GaussianNoise(sigma)(y))

        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    # outputs = Activation('relu')(y)

    mdl_name = 'CNN2D'
    return Model(inputs, outputs, name=mdl_name)



def cnn2d_lnormspat(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
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
    y = Reshape((y.shape[1],y.shape[-2]*y.shape[-1]),dtype=dtype)(y)

    # y = LayerNormalization(axis=[-1,-2],epsilon=1e-7,trainable=False)(y)        # z-score the input across spatiak dimension
    y = LayerNormalization(epsilon=1e-7)(y)        # z-score the input
    y = Reshape((y.shape[1],inputs.shape[-2],inputs.shape[-1]),dtype=dtype)(y)
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)
        
    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  
        
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
            
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
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)

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
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)

        y = Activation('relu')(GaussianNoise(sigma)(y))

        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    # outputs = Activation('relu')(y)

    mdl_name = 'CNN2D_LNORMSPAT'
    return Model(inputs, outputs, name=mdl_name)


def cnn2d_maxpool(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    reg_kern = 1e-4
    reg_act = 1e-4
    
    sigma = 0.01
    
    filt_temporal_width=inputs.shape[1]

    # first layer  
    y = inputs
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(reg_kern))(y)
    
    if MaxPool > 0:
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)
        
    if BatchNorm is True: 
        # y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        rgb = y.shape[1:]
        y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))

    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(reg_kern))(y)                  
        
        if MaxPool > 0:
            y = MaxPool2D(MaxPool,data_format='channels_first')(y)

        if BatchNorm is True: 
            # y = BatchNormalization(axis=1,epsilon=1e-7)(y)
            rgb = y.shape[1:]
            y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))

            
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        if y.shape[-1]<filt3_size:
            filt3_size = (filt3_size,y.shape[-1])
        elif y.shape[-2]<filt3_size:
            filt3_size = (y.shape[-2],filt3_size)
        else:
            filt3_size = filt3_size
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(reg_kern))(y)    
        
        if MaxPool > 0:
            y = MaxPool2D(MaxPool,data_format='channels_first')(y)

        if BatchNorm is True: 
            # y = BatchNormalization(axis=1,epsilon=1e-7)(y)
            rgb = y.shape[1:]
            y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))


        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Fourth layer
    if mdl_params['chan4_n']>0:
        if y.shape[-1]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (mdl_params['filt4_size'],y.shape[-1])
        elif y.shape[-2]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (y.shape[-2],mdl_params['filt4_size'])
        else:
            mdl_params['filt4_size'] = mdl_params['filt4_size']
            
        y = Conv2D(mdl_params['chan4_n'], mdl_params['filt4_size'], data_format="channels_first", kernel_regularizer=l2(reg_kern))(y)    
        
        # if MaxPool > 0:
        #     y = MaxPool2D(MaxPool,data_format='channels_first')(y)

        if BatchNorm is True: 
            # y = BatchNormalization(axis=1,epsilon=1e-7)(y)
            rgb = y.shape[1:]
            y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))


        y = Activation('relu')(GaussianNoise(sigma)(y))

        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(reg_kern), activity_regularizer=l1(reg_act))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'CNN2D_MAXPOOL'
    return Model(inputs, outputs, name=mdl_name)


class trainableAF(tf.keras.layers.Layer):
    def __init__(self,params,units=1,dtype='foat32'):
        super(trainableAF,self).__init__()
        self.units = units
        self.params = params
        # self.dtype='float16'
        

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self,input_shape):
        dtype = self.dtype
        
        s_init = tf.keras.initializers.Constant(self.params['s'])
        self.s = tf.Variable(name='saturation',initial_value=s_init(shape=(1,self.units),dtype=dtype),trainable=self.params['s_trainable'])
        s_scaleFac = tf.keras.initializers.Constant(self.params['s_scaleFac'])
        self.s_scaleFac = tf.Variable(name='s_scaleFac',initial_value=s_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        n_init = tf.keras.initializers.Constant(self.params['n'])
        self.n = tf.Variable(name='gain',initial_value=n_init(shape=(1,self.units),dtype=dtype),trainable=self.params['n_trainable'])
        n_scaleFac = tf.keras.initializers.Constant(self.params['n_scaleFac'])
        self.n_scaleFac = tf.Variable(name='n_scaleFac',initial_value=n_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
       

    def call(self,inputs):
        X_fun = inputs
        
        s = self.s*self.s_scaleFac
        n = self.n*self.n_scaleFac
        
        a = ((1-s)*tf.math.log(1+tf.math.exp(n*X_fun)))/n
        b = (s*(tf.math.exp(n*X_fun)))/(1+tf.math.exp(n*X_fun))
        outputs = a+b

        return outputs
    
def cnn2d_af(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    aru_params = {
        's': 0.5,
        's_scaleFac': 1,
        's_trainable': True,
        'n': 0.1,
        'n_scaleFac': 10,
        'n_trainable': True
        }
    

    reg_kern = 1e-4
    reg_act = 1e-4
    
    sigma = 0.01
    
    filt_temporal_width=inputs.shape[1]

    # first layer  
    y = inputs
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(reg_kern))(y)
    
    if MaxPool > 0:
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)
        
    if BatchNorm is True: 
        # y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        rgb = y.shape[1:]
        y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))

    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(reg_kern))(y)                  
        
        if MaxPool > 0:
            y = MaxPool2D(MaxPool,data_format='channels_first')(y)

        if BatchNorm is True: 
            # y = BatchNormalization(axis=1,epsilon=1e-7)(y)
            rgb = y.shape[1:]
            y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))

            
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        if y.shape[-1]<filt3_size:
            filt3_size = (filt3_size,y.shape[-1])
        elif y.shape[-2]<filt3_size:
            filt3_size = (y.shape[-2],filt3_size)
        else:
            filt3_size = filt3_size
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(reg_kern))(y)    
        
        if MaxPool > 0:
            y = MaxPool2D(MaxPool,data_format='channels_first')(y)

        if BatchNorm is True: 
            # y = BatchNormalization(axis=1,epsilon=1e-7)(y)
            rgb = y.shape[1:]
            y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))


        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Fourth layer
    if mdl_params['chan4_n']>0:
        if y.shape[-1]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (mdl_params['filt4_size'],y.shape[-1])
        elif y.shape[-2]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (y.shape[-2],mdl_params['filt4_size'])
        else:
            mdl_params['filt4_size'] = mdl_params['filt4_size']
            
        y = Conv2D(mdl_params['chan4_n'], mdl_params['filt4_size'], data_format="channels_first", kernel_regularizer=l2(reg_kern))(y)    
        
        # if MaxPool > 0:
        #     y = MaxPool2D(MaxPool,data_format='channels_first')(y)

        if BatchNorm is True: 
            # y = BatchNormalization(axis=1,epsilon=1e-7)(y)
            rgb = y.shape[1:]
            y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))


        y = Activation('relu')(GaussianNoise(sigma)(y))

        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(reg_kern), activity_regularizer=l1(reg_act))(y)
    # outputs = Activation('softplus')(y)
    outputs = trainableAF(aru_params,units=y.shape[-1])(y)


    mdl_name = 'CNN2D_AF'
    return Model(inputs, outputs, name=mdl_name)
# %% PR models

@tf.function(autograph=True,experimental_relax_shapes=True)
def riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark):
    darkCurrent = gdark**cgmphill * cgmp2cur/2
    gdark = (2 * darkCurrent / cgmp2cur) **(1/cgmphill)
    
    cur2ca = beta * cdark / darkCurrent                # get q using steady state
    smax = eta/phi * gdark * (1 + (cdark / hillaffinity) **hillcoef)		# get smax using steady state
    
    n_spatialDims = X_fun.shape[-1]
    tme = tf.range(0,X_fun.shape[1],dtype=X_fun.dtype)*TimeStep
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
    def __init__(self,pr_params,units=1,dtype='foat16'):
        super(photoreceptor_REIKE,self).__init__()
        self.units = units
        self.pr_params = pr_params
        # self.dtype='float16'
        

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self,input_shape):
        dtype = self.dtype
        
        sigma_init = tf.keras.initializers.Constant(self.pr_params['sigma'])
        self.sigma = tf.Variable(name='sigma',initial_value=sigma_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['sigma_trainable'])
        sigma_scaleFac = tf.keras.initializers.Constant(self.pr_params['sigma_scaleFac'])
        self.sigma_scaleFac = tf.Variable(name='sigma_scaleFac',initial_value=sigma_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        phi_init = tf.keras.initializers.Constant(self.pr_params['phi'])
        self.phi = tf.Variable(name='phi',initial_value=phi_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['phi_trainable'])
        phi_scaleFac = tf.keras.initializers.Constant(self.pr_params['phi_scaleFac'])
        self.phi_scaleFac = tf.Variable(name='phi_scaleFac',initial_value=phi_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
       
        eta_init = tf.keras.initializers.Constant(self.pr_params['eta'])
        self.eta = tf.Variable(name='eta',initial_value=eta_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['eta_trainable'])
        eta_scaleFac = tf.keras.initializers.Constant(self.pr_params['eta_scaleFac'])
        self.eta_scaleFac = tf.Variable(name='eta_scaleFac',initial_value=eta_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        beta_init = tf.keras.initializers.Constant(self.pr_params['beta'])
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['beta_trainable'])
        beta_scaleFac = tf.keras.initializers.Constant(self.pr_params['beta_scaleFac'])
        self.beta_scaleFac = tf.Variable(name='beta_scaleFac',initial_value=beta_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)

        cgmp2cur_init = tf.keras.initializers.Constant(self.pr_params['cgmp2cur'])
        self.cgmp2cur = tf.Variable(name='cgmp2cur',initial_value=cgmp2cur_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['cgmp2cur_trainable'])
        
        cgmphill_init = tf.keras.initializers.Constant(self.pr_params['cgmphill'])
        self.cgmphill = tf.Variable(name='cgmphill',initial_value=cgmphill_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['cgmphill_trainable'])
        cgmphill_scaleFac = tf.keras.initializers.Constant(self.pr_params['cgmphill_scaleFac'])
        self.cgmphill_scaleFac = tf.Variable(name='cgmphill_scaleFac',initial_value=cgmphill_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        cdark_init = tf.keras.initializers.Constant(self.pr_params['cdark'])
        self.cdark = tf.Variable(name='cdark',initial_value=cdark_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['cdark_trainable'])
        
        betaSlow_init = tf.keras.initializers.Constant(self.pr_params['betaSlow'])
        self.betaSlow = tf.Variable(name='betaSlow',initial_value=betaSlow_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['betaSlow_trainable'])
        betaSlow_scaleFac = tf.keras.initializers.Constant(self.pr_params['betaSlow_scaleFac'])
        self.betaSlow_scaleFac = tf.Variable(name='betaSlow_scaleFac',initial_value=betaSlow_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        hillcoef_init = tf.keras.initializers.Constant(self.pr_params['hillcoef'])
        self.hillcoef = tf.Variable(name='hillcoef',initial_value=hillcoef_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['hillcoef_trainable'])
        hillcoef_scaleFac = tf.keras.initializers.Constant(self.pr_params['hillcoef_scaleFac'])
        self.hillcoef_scaleFac = tf.Variable(name='hillcoef_scaleFac',initial_value=hillcoef_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        hillaffinity_init = tf.keras.initializers.Constant(self.pr_params['hillaffinity'])
        self.hillaffinity = tf.Variable(name='hillaffinity',initial_value=hillaffinity_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['hillaffinity_trainable'])
        hillaffinity_scaleFac = tf.keras.initializers.Constant(self.pr_params['hillaffinity_scaleFac'])
        self.hillaffinity_scaleFac = tf.Variable(name='hillaffinity_scaleFac',initial_value=hillaffinity_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        gamma_init = tf.keras.initializers.Constant(self.pr_params['gamma'])
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['gamma_trainable'])
        gamma_scaleFac = tf.keras.initializers.Constant(self.pr_params['gamma_scaleFac'])
        self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
                
        gdark_init = tf.keras.initializers.Constant(self.pr_params['gdark'])
        self.gdark = tf.Variable(name='gdark',initial_value=gdark_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['gdark_trainable'])
        gdark_scaleFac = tf.keras.initializers.Constant(self.pr_params['gdark_scaleFac'])
        self.gdark_scaleFac = tf.Variable(name='gdark_scaleFac',initial_value=gdark_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)


 
    def call(self,inputs):
        X_fun = inputs

        timeBin = float(self.pr_params['timeBin']) # ms
        frameTime = timeBin # ms
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


def prfr_cnn2d(inputs,n_out,**kwargs): #(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
    filt_temporal_width=kwargs['filt_temporal_width']

    pr_params = kwargs['pr_params']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.1
    
    y = inputs
    # y = BatchNormalization(axis=-3,epsilon=1e-7)(y)
    y = Reshape((y.shape[1],y.shape[-2]*y.shape[-1]),dtype=dtype)(y)
    y = photoreceptor_REIKE(pr_params,units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y = LayerNormalization(axis=-3,epsilon=1e-7)(y)      # Along the temporal axis

    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
        
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)

    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)

    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRFR_CNN2D'
    return Model(inputs, outputs, name=mdl_name)



def prfr_cnn2d_noln(inputs,n_out,**kwargs): #(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
    filt_temporal_width=kwargs['filt_temporal_width']

    pr_params = kwargs['pr_params']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.1
    
    y = inputs
    y = Reshape((y.shape[1],y.shape[-2]*y.shape[-1]),dtype=dtype)(y)
    y = photoreceptor_REIKE(pr_params,units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    # y = LayerNormalization(axis=-3,epsilon=1e-7)(y)      # Along the temporal axis

    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
        
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)

    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)

    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRFR_CNN2D_NOLN'
    return Model(inputs, outputs, name=mdl_name)



# %% Linear cone model

def generate_linear_filter(alpha,tau_r,tau_d,tau_o,omega,t):
    a = alpha*(((t/tau_r)**4)/(1+(t/tau_r)**4))
    b = tf.math.exp(-1*(t/tau_d))
    c = tf.math.cos(((2*3.142*t)/tau_o)+omega)
    
    kern = a*b*c

    return kern


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

class photoreceptor_LINEAR(tf.keras.layers.Layer):
    def __init__(self,pr_params,units=1):
        super(photoreceptor_LINEAR,self).__init__()
        self.units = units
        self.pr_params = pr_params

            
    def build(self,input_shape):
        
        dtype=tf.float32
        
        alpha_init = tf.keras.initializers.Constant(self.pr_params['alpha'])
        self.alpha = tf.Variable(name='alpha',initial_value=alpha_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['alpha_trainable'])
        alpha_scaleFac = tf.keras.initializers.Constant(self.pr_params['alpha_scaleFac'])
        self.alpha_scaleFac = tf.Variable(name='alpha_scaleFac',initial_value=alpha_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        tau_r_init = tf.keras.initializers.Constant(self.pr_params['tau_r'])
        self.tau_r = tf.Variable(name='tau_r',initial_value=tau_r_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['tau_r_trainable'])
        tau_r_scaleFac = tf.keras.initializers.Constant(self.pr_params['tau_r_scaleFac'])
        self.tau_r_scaleFac = tf.Variable(name='tau_r_scaleFac',initial_value=tau_r_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
    
    
        tau_d_init = tf.keras.initializers.Constant(self.pr_params['tau_d'])
        self.tau_d = tf.Variable(name='tau_d',initial_value=tau_d_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['tau_d_trainable'])
        tau_d_scaleFac = tf.keras.initializers.Constant(self.pr_params['tau_d_scaleFac'])
        self.tau_d_scaleFac = tf.Variable(name='tau_d_scaleFac',initial_value=tau_d_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)


        tau_o_init = tf.keras.initializers.Constant(self.pr_params['tau_o'])
        self.tau_o = tf.Variable(name='tau_o',initial_value=tau_o_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['tau_o_trainable'])
        tau_o_scaleFac = tf.keras.initializers.Constant(self.pr_params['tau_o_scaleFac'])
        self.tau_o_scaleFac = tf.Variable(name='tau_o_scaleFac',initial_value=tau_o_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)

        omega_init = tf.keras.initializers.Constant(self.pr_params['omega'])
        self.omega = tf.Variable(name='omega',initial_value=omega_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['omega_trainable'])
        omega_scaleFac = tf.keras.initializers.Constant(self.pr_params['omega_scaleFac'])
        self.omega_scaleFac = tf.Variable(name='omega_scaleFac',initial_value=omega_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)


    
    def call(self,inputs):
       
        timeBin = 8
        
        alpha =  self.alpha / timeBin
        tau_r = self.tau_r / timeBin
        tau_d = self.tau_d / timeBin
        tau_o = self.tau_o / timeBin
        omega = self.omega / timeBin

        
        t = tf.range(0,1000/timeBin,dtype='float32')
        
        kernel_1D = generate_linear_filter(alpha,tau_r,tau_d,tau_o,omega,t)          
        y_tf = conv_oper(inputs,kernel_1D)
   
        outputs = y_tf
        
        return outputs
    


def prfr_ln_cnn2d(inputs,n_out,**kwargs): #(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
    filt_temporal_width=kwargs['filt_temporal_width']

    pr_params = kwargs['pr_params']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.1
    
    y = inputs
    # y = BatchNormalization(axis=-3,epsilon=1e-7)(y)
    y = Reshape((y.shape[1],y.shape[-2]*y.shape[-1]),dtype=dtype)(y)
    y = photoreceptor_LINEAR(pr_params,units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y = LayerNormalization(axis=-3,epsilon=1e-7)(y)      # Along the temporal axis

    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
        
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)

    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)

    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRFR_LN_CNN2D'
    return Model(inputs, outputs, name=mdl_name)


# %% Clark model

def generate_simple_filter(tau,n,t):
    f = (t**n)*tf.math.exp(-t/tau) # functional form in paper
    f = (f/tau**(n+1))/tf.math.exp(tf.math.lgamma(n+1)) # normalize appropriately
    # print(t.shape)
    # print(n.shape)
    # print(tau.shape)

    return f

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
def prda_cnn2d(inputs,n_out,**kwargs): #(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
    filt_temporal_width=kwargs['filt_temporal_width']

    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.1
    
    y = inputs
    # y = BatchNormalization(axis=-3,epsilon=1e-7)(y)
    y = Reshape((y.shape[1],y.shape[-2]*y.shape[-1]),dtype=dtype)(y)
    y = photoreceptor_DA(units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y = LayerNormalization(axis=-3,epsilon=1e-7)(y)      # Along the temporal axis

    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
        
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)

    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)

    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRDA_CNN2D'
    return Model(inputs, outputs, name=mdl_name)


# %% Adaptive RGC

class rgc_adaptive(tf.keras.layers.Layer):
    def __init__(self,params,units=1,dtype='foat32'):
        super(rgc_adaptive,self).__init__()
        self.units = units
        self.params = params
        # self.dtype='float16'
        

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self,input_shape):
        dtype = self.dtype
        
        s_init = tf.keras.initializers.Constant(self.params['s'])
        self.s = tf.Variable(name='saturation',initial_value=s_init(shape=(1,self.units),dtype=dtype),trainable=self.params['s_trainable'])
        s_scaleFac = tf.keras.initializers.Constant(self.params['s_scaleFac'])
        self.s_scaleFac = tf.Variable(name='s_scaleFac',initial_value=s_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        n_init = tf.keras.initializers.Constant(self.params['n'])
        self.n = tf.Variable(name='gain',initial_value=n_init(shape=(1,self.units),dtype=dtype),trainable=self.params['n_trainable'])
        n_scaleFac = tf.keras.initializers.Constant(self.params['n_scaleFac'])
        self.n_scaleFac = tf.Variable(name='n_scaleFac',initial_value=n_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
       

    def call(self,inputs):
        X_fun = inputs
        
        s = self.s*self.s_scaleFac
        n = self.n*self.n_scaleFac
        
        a = ((1-s)*tf.math.log(1+tf.math.exp(n*X_fun)))/n
        b = (s*(tf.math.exp(n*X_fun)))/(1+tf.math.exp(n*X_fun))
        outputs = a+b

        return outputs


def prfr_cnn2d_aru(inputs,n_out,**kwargs): #(inputs,n_out,filt_temporal_width=120,chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
    filt_temporal_width=kwargs['filt_temporal_width']

    pr_params = kwargs['pr_params']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.1
    
    y = inputs
    # y = BatchNormalization(axis=-3,epsilon=1e-7)(y)
    y = Reshape((y.shape[1],y.shape[-2]*y.shape[-1]),dtype=dtype)(y)
    y = photoreceptor_REIKE(pr_params,units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y = LayerNormalization(axis=-3,epsilon=1e-7)(y)      # Along the temporal axis

    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
        
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)

    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    # outputs = Activation('softplus',dtype='float32')(y)
    aru_params = {
        's': 0.5,
        's_scaleFac': 1,
        's_trainable': True,
        'n': 0.1,
        'n_scaleFac': 10,
        'n_trainable': True
        }
    
    outputs = rgc_adaptive(aru_params,units=y.shape[-1])(y)

    mdl_name = 'PRFR_CNN2D_ARU'
    return Model(inputs, outputs, name=mdl_name)