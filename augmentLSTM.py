#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 08:56:30 2021

@author: saad
"""

import numpy as np
import os
import math
import csv

import tensorflow as tf
config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = .9
tf.compat.v1.Session(config=config)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
  
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

from model.data_handler import load_h5Dataset, prepare_data_cnn3d, rolling_window
from model.performance import save_modelPerformance
import model.metrics as metrics
from model.models import cnn_3d, LSTM_alone
from model.train_model import train
from model.load_savedModel import load

from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])

import matplotlib.pyplot as plt


import gc


import datetime

# %%

expDate = '20180502_s3'
temporal_width=60
thresh_rr=0.45

chan1_n=13
filt1_size=11
filt1_3rdDim=40

chan2_n=25
filt2_size=7
filt2_3rdDim=5

chan3_n=25
filt3_size=3
filt3_3rdDim=6

nb_epochs=100
bz_ms=10000
    
    
path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_saad',expDate,'datasets')
path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'
path_model_save_base = os.path.join('/home/saad/data/analyses/data_saad',expDate)


fname_data_train_val_test = os.path.join(path_dataset,(expDate+"_dataset_train_val_test.h5"))
data_train,data_val,data_test,data_quality,dataset_rr,parameters = load_h5Dataset(fname_data_train_val_test)
    
idx_unitsToTake = data_quality['dist_cc']>thresh_rr

data_train = prepare_data_cnn3d(data_train,temporal_width,idx_unitsToTake)
data_test = prepare_data_cnn3d(data_test,temporal_width,idx_unitsToTake)
data_val = prepare_data_cnn3d(data_val,temporal_width,idx_unitsToTake)

t_frame = parameters['t_frame']
unit_noise = data_quality['var_noise_dset_all'][idx_unitsToTake]

# retinal reliability
fractionExplainableVariance_allUnits = (data_quality['fractionExplainableVariance_allUnits'][idx_unitsToTake] - unit_noise) / data_quality['fractionExplainableVariance_allUnits'][idx_unitsToTake]
retinalReliability = np.round(np.nanmedian(fractionExplainableVariance_allUnits),2)

# Model params
BatchNorm = True
MaxPool = True

bz = math.ceil(bz_ms/t_frame)


fname_model = 'T-%03d_C1-%02d-%02d-%02d_C2-%02d-%02d-%02d_C3-%02d-%02d-%02d' %(temporal_width,chan1_n,filt1_size,filt1_3rdDim,
                                                                             chan2_n,filt2_size,filt2_3rdDim,
                                                                             chan3_n,filt3_size,filt3_3rdDim)


# %%
mdl_name = 'CNN_3D'
path_model_save = os.path.join(path_model_save_base,mdl_name,fname_model)
path_save_model_performance = os.path.join(path_model_save,'performance')
if not os.path.exists(path_save_model_performance):
    os.makedirs(path_save_model_performance)


counter_train = 0
x = Input(shape=data_train.X.shape[1:])
n_cells = data_train.y.shape[1]

# mdl = cnn_3d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm)

mdl = load(os.path.join(path_model_save,fname_model))


# %% LSTM
layer_name = 'flatten_3'
layer_output = mdl.get_layer(layer_name).output
new_model = Model(mdl.input, outputs=layer_output)
cnn_output_train = new_model.predict(data_train.X)
cnn_output_val = new_model.predict(data_val.X)
_ = gc.collect()
#%%
# % prepare training and validation data for LSTM layers
lstm_timeStep = 240
bz_ms = 10000

X = rolling_window(cnn_output_train,lstm_timeStep,time_axis=0)
y = data_train.y[lstm_timeStep:]
data_train_lstm = Exptdata(X,y)

X = rolling_window(cnn_output_val,lstm_timeStep,time_axis=0)
y = data_val.y[lstm_timeStep:]
data_val_lstm = Exptdata(X,y)

del X, y
# % Train LSTM layer based on pretrained CNN
nb_epochs = 40


mdl_select = 'augmentedLSTM'
x = Input(shape=data_train_lstm.X.shape[1:])
n_cells = data_train_lstm.y.shape[1]

fname_model_lstm = mdl_select+'_T-%03d_B-%d' %(lstm_timeStep,bz_ms)
fname_excel = 'performance_'+mdl_select+'.csv'


path_model_save = os.path.join(path_model_save_base,mdl_select,fname_model_lstm)
if not os.path.exists(path_model_save):
    os.makedirs(path_model_save)


mdl_lstm = LSTM_alone(x, n_cells, lstm_timeStep, BatchNorm=True)
mdl_lstm_history = train(mdl_lstm, data_train_lstm, data_val_lstm,fname_excel,counter_train,path_model_save, fname_model_lstm, bz, nb_epochs=nb_epochs,validation_batch_size = bz,validation_freq=150)

mdl_lstm_history = mdl_lstm_history.history

# %

obs_rate = data_val_lstm.y
fev_median_allEpochs = np.empty(nb_epochs)
fev_median_allEpochs[:] = np.nan
fev_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
fev_allUnits_allEpochs[:] = np.nan

print('-----EVALUATING PERFORMANCE-----')
for i in range(nb_epochs-1):
    weight_file = 'weights_'+fname_model_lstm+'_epoch-%03d.h5' % (i+1)
    mdl_lstm.load_weights(os.path.join(path_model_save,weight_file))
    pred_rate = mdl_lstm.predict(data_val_lstm.X)
    rgb = metrics.fraction_of_explainable_variance_explained(obs_rate,pred_rate,unit_noise)
    fev_allUnits_allEpochs[i,:] = rgb
    fev_median_allEpochs[i] = np.nanmedian(rgb)
    _ = gc.collect()


idx_bestEpoch = np.nanargmax(fev_median_allEpochs)
fev_median_bestEpoch = np.round(fev_median_allEpochs[idx_bestEpoch],2)
fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
fname_bestWeight = 'weights_'+fname_model_lstm+'_epoch-%03d.h5' % (idx_bestEpoch+1)
mdl_lstm.load_weights(os.path.join(path_model_save,fname_bestWeight))
pred_rate = mdl_lstm.predict(data_val_lstm.X)
_ = gc.collect()


# %% Save performance
print('-----SAVING PERFORMANCE STUFF TO H5-----')

fname_save_performance = os.path.join(path_save_model_performance,(expDate+'_'+fname_model+'.h5'))

model_performance = {
    'fev_median_allEpochs': fev_median_allEpochs,
    'fev_allUnits_allEpochs': fev_allUnits_allEpochs,
    'idx_bestEpoch': idx_bestEpoch,
    'fev_median_bestEpoch': fev_median_bestEpoch,
    'fev_allUnits_bestEpoch': fev_allUnits_bestEpoch,
    'fname_bestWeight': fname_bestWeight,
    'fractionExplainableVariance_allUnits': fractionExplainableVariance_allUnits,
    'retinalReliability': retinalReliability,
    }

metaInfo = {
   ' mdl_name': mdl.name,
    'path_model_save': path_model_save,
    'uname_selectedUnits': np.array(data_quality['uname_selectedUnits'][idx_unitsToTake],dtype='bytes'),
    'idx_unitsToTake': idx_unitsToTake,
    'thresh_rr': thresh_rr,
    'N_TRIALS': counter_train+1,
    'Date': np.array(datetime.datetime.now(),dtype='bytes')
    }
    
model_params = {
            'chan1_n' : chan1_n,
            'filt1_size' : filt1_size,
            'filt1_3rdDim': filt1_3rdDim,
            'chan2_n' : chan2_n,
            'filt2_size' : filt2_size,
            'filt2_3rdDim': filt2_3rdDim,
            'chan3_n' : chan3_n,
            'filt3_size' : filt3_size,
            'filt3_3rdDim': filt3_3rdDim,            
            'bz_ms' : bz_ms,
            'nb_epochs' : nb_epochs,
            'BatchNorm': BatchNorm,
            'MaxPool': MaxPool,
            }

stim_info = {
     'fname_data_train_val_test':fname_data_train_val_test,
     'n_trainingSamps': data_train.X.shape[0],
     'n_valSamps': data_val.X.shape[0],
     'n_testSamps': data_test.X.shape[0],
     'temporal_width':temporal_width,
     }

datasets_val = {
    'data_val_X': data_val.X,
    'data_val_y': data_val.y,
    'data_test_X': data_test.X,
    'data_test_y': data_test.y,
    }

dataset_pred = {
    'obs_rate': obs_rate,
    'pred_rate': pred_rate,
    }

save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred)

# %% Write performance to csv file
print('-----WRITING TO CSV FILE-----')
csv_header = ['arg_unit','RR','Temp_filt','batch_size','epochs','chans','filt','temp','chans','filt','temp','chans','filt','temp','FEV_median']
csv_data = [thresh_rr,retinalReliability,temporal_width,bz_ms,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,fev_median_bestEpoch]

fname_csv_file = 'performance_'+expDate+'_'+mdl.name+'.csv'
fname_csv_file = os.path.join(path_save_performance,fname_csv_file)
if not os.path.exists(fname_csv_file):
    with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(csv_header) 
        
with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(csv_data) 


fname_validation_excel = os.path.join(path_save_model_performance,expDate+'_validation_'+fname_model+'.csv')
csv_header = ['epoch','val_fev']
with open(fname_validation_excel,'w',encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(csv_header) 
    
    for i in range(fev_median_allEpochs.shape[0]):
        csvwriter.writerow([str(i),str(np.round(fev_median_allEpochs[i],2))]) 
        
    


print('-----FINISHED-----')

