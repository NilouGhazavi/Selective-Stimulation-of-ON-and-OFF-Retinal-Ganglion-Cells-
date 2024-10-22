#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 21:00:57 2021

@author: saad
"""
import os
import h5py

import tensorflow as tf
import tensorflow.keras.callbacks as cb
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import model.metrics as metrics
import numpy as np
import re


def chunker(data,batch_size,nsamps=0):
    if nsamps == 0:
        X = data.X
        y = data.y
    else:
        X = data.X[:nsamps]
        y = data.y[:nsamps]
        
    counter = 0
    while True:
        counter = (counter + 1) % X.shape[0]
        for cbatch in range(0, X.shape[0], batch_size):
            yield (X[cbatch:(cbatch + batch_size)], y[cbatch:(cbatch + batch_size)])

 
def train(mdl, data_train, data_val,fname_excel,path_model_base, fname_model, bz=588, nb_epochs=200, validation_batch_size=5000,validation_freq=10,USE_CHUNKER=0,initial_epoch=1):
    lr = 1e-2

    mdl.compile(loss='poisson', optimizer=Adam(lr), metrics=[metrics.cc, metrics.rmse, metrics.fev])
    
    if initial_epoch>1:
        try:
            weight_file = 'weights_'+fname_model+'_epoch-%03d' % initial_epoch
            mdl.load_weights(os.path.join(path_model_base,weight_file))
        except:
            weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % initial_epoch
            mdl.load_weights(os.path.join(path_model_base,weight_file))


    # define model callbacks
    fname_cb = 'weights_'+ fname_model + '_epoch-{epoch:03d}' 
    cbs = [cb.ModelCheckpoint(os.path.join(path_model_base, fname_cb),save_weights_only=True),
           cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
           cb.TensorBoard(log_dir=path_model_base, histogram_freq=0, write_grads=False),
           cb.CSVLogger(os.path.join(path_model_base, fname_excel))]
    
    # cbs = [cb.ModelCheckpoint(os.path.join(path_model_base, fname_cb)),
    #        cb.TensorBoard(log_dir=path_model_base, histogram_freq=0, write_grads=False),
    #        cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
    #        cb.CSVLogger(os.path.join(path_model_base, fname_excel))]

        
        # try:
        #     weight_file = 'weights_'+fname_model+'_epoch-%03d' % initial_epoch
        # except:
        #     weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % initial_epoch
            
        # mdl.load_weights(os.path.join(path_model_save,weight_file))


    if USE_CHUNKER==0:
        mdl_history = mdl.fit(x=data_train.X, y=data_train.y, batch_size=bz, epochs=nb_epochs,
                          callbacks=cbs, validation_data=(data_val.X,data_val.y), validation_batch_size=validation_batch_size, validation_freq=validation_freq, shuffle=True, initial_epoch=initial_epoch,use_multiprocessing=True)    # validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val)
        
    else:
        batch_size = bz
        steps_per_epoch = int(np.ceil(data_train.X.shape[0]/batch_size))
        gen = chunker(data_train,batch_size,0)
        mdl_history = mdl.fit(gen,steps_per_epoch=steps_per_epoch,epochs=nb_epochs,callbacks=cbs, validation_data=(data_val.X,data_val.y), shuffle=True,initial_epoch=initial_epoch,use_multiprocessing=True)    # validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val) # steps_per_epoch=steps_per_epoch

    # mdl_history = mdl.fit(x=data_train.X, y=data_train.y, batch_size=bz, epochs=nb_epochs,
    #                   callbacks=cbs, shuffle=True)    # validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val)

      
    rgb = mdl_history.history
    keys = list(rgb.keys())
    
    fname_history = 'history_'+mdl.name+'.h5'
    fname_history = os.path.join(path_model_base,fname_history)                            
    f = h5py.File(fname_history,'w')
    for i in range(len(rgb)):
        f.create_dataset(keys[i], data=rgb[keys[i]])
    f.close()
    
    
    # fname_model = mdl.name+'_%02d' %training_trial
    # fname_model = fname_cb_prefix
    mdl.save(os.path.join(path_model_base,fname_model))
    
    return mdl_history

