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
import h5py

import tensorflow as tf
config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = .9
tf.compat.v1.Session(config=config)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
  
from tensorflow.keras.layers import Input

from model.data_handler import load_h5Dataset, prepare_data_cnn3d, prepare_data_cnn2d, prepare_data_convLSTM, prepare_data_pr_cnn2d
from model.performance import save_modelPerformance, model_evaluate, model_evaluate_new
import model.metrics as metrics
from model.models import cnn_3d, cnn_2d, pr_cnn2d, pr_cnn3d, prfr_cnn2d_fixed
from model.train_model import train
from model.load_savedModel import load
from tensorflow.keras.optimizers import Adam
 

import gc


import datetime

# %%
def run_fixPerformance(expDate,mdl_name,path_model_save_base,name_datasetFile,fname_performance_excel,samps_shift=4,saveToCSV=1,runOnCluster=0,
                            temporal_width=40, pr_temporal_width = 180, thresh_rr=0,
                            chan1_n=8, filt1_size=13, filt1_3rdDim=20,
                            chan2_n=0, filt2_size=0, filt2_3rdDim=0,
                            chan3_n=0, filt3_size=0, filt3_3rdDim=0,
                            nb_epochs=100,bz_ms=10000,BatchNorm=1,MaxPool=1,c_trial=1,BatchNorm_train=0,idx_CNN_start=1,
                            path_dataset_base='/home/saad/data/analyses/data_kiersten',path_existing_mdl=''):

# %%    
# expDate = 'retina1'
# mdl_name = 'CNN_2D'

# runOnCluster=0
# temporal_width=60
# thresh_rr=0.15
# chan1_n=13
# filt1_size=1
# filt1_3rdDim=0
# chan2_n=13
# filt2_size=3
# filt2_3rdDim=0
# chan3_n=25
# filt3_size=3
# filt3_3rdDim=0
# nb_epochs=150
# bz_ms=10000
# BatchNorm=1
# MaxPool=0    
# c_trial = 1
    
# path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_saad',expDate,'datasets')
# path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'
# path_model_save_base = os.path.join('/home/saad/data/analyses/data_saad',expDate)


# load train val and test datasets from saved h5 file
    path_dataset = os.path.join(path_dataset_base,'datasets')
    path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'
    # path_model_save_base = os.path.join('/home/saad/data/analyses/data_kiersten',expDate)
    
    
    # load train val and test datasets from saved h5 file
    fname_data_train_val_test = os.path.join(path_dataset,name_datasetFile)
    data_train,data_val,data_test,data_quality,dataset_rr,parameters,_ = load_h5Dataset(fname_data_train_val_test)

        
    # Arrange data according to needs
    idx_unitsToTake = data_quality['idx_unitsToTake']
    idx_unitsToTake
    temporal_width_eval = temporal_width
    
    if mdl_name == 'CNN_3D' or mdl_name == 'CNN_3D_INCEP' or mdl_name == 'CNN_3D_LSTM':
        data_train = prepare_data_cnn3d(data_train,temporal_width,np.arange(len(idx_unitsToTake)))
        data_test = prepare_data_cnn3d(data_test,temporal_width,np.arange(len(idx_unitsToTake)))
        data_val = prepare_data_cnn3d(data_val,temporal_width,np.arange(len(idx_unitsToTake)))
    
    elif mdl_name == 'PR_CNN3D':
       data_train = prepare_data_cnn3d(data_train,pr_temporal_width,np.arange(len(idx_unitsToTake)))
       data_test = prepare_data_cnn3d(data_test,pr_temporal_width,np.arange(len(idx_unitsToTake)))
       data_val = prepare_data_cnn3d(data_val,pr_temporal_width,np.arange(len(idx_unitsToTake)))
       temporal_width_eval = pr_temporal_width
       
    elif mdl_name == 'CNN_2D' or mdl_name=='CNN_2D_LSTM':
        data_train = prepare_data_cnn2d(data_train,temporal_width,np.arange(len(idx_unitsToTake)))
        data_test = prepare_data_cnn2d(data_test,temporal_width,np.arange(len(idx_unitsToTake)))
        data_val = prepare_data_cnn2d(data_val,temporal_width,np.arange(len(idx_unitsToTake)))       
    
    elif mdl_name == 'convLSTM' or mdl_name == 'LSTM_CNN_2D':
        data_train = prepare_data_convLSTM(data_train,temporal_width,np.arange(len(idx_unitsToTake)))
        data_test = prepare_data_convLSTM(data_test,temporal_width,np.arange(len(idx_unitsToTake)))
        data_val = prepare_data_convLSTM(data_val,temporal_width,np.arange(len(idx_unitsToTake)))   
    
    elif mdl_name == 'PR_CNN2D':
        data_train = prepare_data_pr_cnn2d(data_train,pr_temporal_width,np.arange(len(idx_unitsToTake)))
        data_test = prepare_data_pr_cnn2d(data_test,pr_temporal_width,np.arange(len(idx_unitsToTake)))
        data_val = prepare_data_pr_cnn2d(data_val,pr_temporal_width,np.arange(len(idx_unitsToTake)))
        temporal_width_eval = pr_temporal_width
        
    elif mdl_name[:10] == 'PRFR_CNN2D':
        data_train = prepare_data_cnn2d(data_train,pr_temporal_width,np.arange(len(idx_unitsToTake)))
        data_test = prepare_data_cnn2d(data_test,pr_temporal_width,np.arange(len(idx_unitsToTake)))
        data_val = prepare_data_cnn2d(data_val,pr_temporal_width,np.arange(len(idx_unitsToTake)))
        temporal_width_eval = pr_temporal_width


    
    t_frame = parameters['t_frame']
    
        
    
    
    if BatchNorm:
            bn_val=1
            BatchNorm=True
    else:
        bn_val=0
        BatchNorm=False
    if MaxPool:
        mp_val=1
        MaxPool=True
    else:
        mp_val=0       
        MaxPool=False
        
    BatchNorm_train = False
     
    bz = math.ceil(bz_ms/t_frame)
     
    x = Input(shape=data_train.X.shape[1:])
    n_cells = data_train.y.shape[1]
       
    if mdl_name == 'CNN_3D':       
        mdl = cnn_3d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d-%02d_C2-%02d-%02d-%02d_C3-%02d-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,filt1_3rdDim,
                                                                                     chan2_n,filt2_size,filt2_3rdDim,
                                                                                     chan3_n,filt3_size,filt3_3rdDim,
                                                                                     bn_val,mp_val,c_trial)
    elif mdl_name=='CNN_2D':
        mdl = cnn_2d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool,BatchNorm_train = BatchNorm_train)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,c_trial)
        filt1_3rdDim=0
        filt2_3rdDim=0
        filt3_3rdDim=0
        
    elif mdl_name=='PR_CNN2D':
        mdl = pr_cnn2d(x, n_cells, filt_temporal_width = temporal_width, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool,BatchNorm_train = BatchNorm_train)
        fname_model = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,pr_temporal_width,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,c_trial)
        filt1_3rdDim=0
        filt2_3rdDim=0
        filt3_3rdDim=0

    elif mdl_name=='PR_CNN3D':
        mdl = pr_cnn3d(x, n_cells, filt_temporal_width = temporal_width, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool,BatchNorm_train = BatchNorm_train)
        fname_model = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,pr_temporal_width,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,c_trial)

        
    elif mdl_name=='PRFR_CNN2D_fixed': # freds model
        rgb = os.path.split(path_existing_mdl)[-1]
        mdl_existing = load(os.path.join(path_existing_mdl,rgb))
        # idx_CNN_start = 5
        

        mdl = prfr_cnn2d_fixed(mdl_existing,idx_CNN_start,x, n_cells, filt_temporal_width=temporal_width,
                             chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size,
                             BatchNorm=BatchNorm,MaxPool=MaxPool,BatchNorm_train = BatchNorm_train)
        
        fname_model = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,pr_temporal_width,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,c_trial)
        filt1_3rdDim=0
        filt2_3rdDim=0
        filt3_3rdDim=0


    else:
        raise ValueError('Wrong model name')    
    path_model_save = path_model_save = os.path.join(path_model_save_base,mdl_name,fname_model)
    path_save_model_performance = os.path.join(path_model_save,'performance')
    if not os.path.exists(path_save_model_performance):
        os.mkdir(path_save_model_performance)
        
    
    # fname_excel = 'performance_'+fname_model+'_chansVary_newFEV.csv'
    
    
    #%% Evaluate performance of the model
    nb_epochs = len([f for f in os.listdir(path_model_save) if f.endswith('index')])
    if nb_epochs == 0:
        nb_epochs = len([f for f in os.listdir(path_model_save) if f.startswith('weights')])
    
    x = Input(shape=data_train.X.shape[1:])
    n_cells = data_train.y.shape[1]
    lr = 1e-2
    
    # mdl = cnn_3d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm)
    
    mdl = load(os.path.join(path_model_save,fname_model))
    # mdl.compile(loss='poisson', optimizer=Adam(lr), metrics=[metrics.cc, metrics.rmse, metrics.fev])
    
    obs_rate = data_val.y
    val_loss_allEpochs = np.empty(nb_epochs)
    val_loss_allEpochs[:] = np.nan
    fev_medianUnits_allEpochs = np.empty(nb_epochs)
    fev_medianUnits_allEpochs[:] = np.nan
    fev_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    fev_allUnits_allEpochs[:] = np.nan
    fracExVar_medianUnits_allEpochs = np.empty(nb_epochs)
    fracExVar_medianUnits_allEpochs[:] = np.nan
    fracExVar_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    fracExVar_allUnits_allEpochs[:] = np.nan
    
    predCorr_medianUnits_allEpochs = np.empty(nb_epochs)
    predCorr_medianUnits_allEpochs[:] = np.nan
    predCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    predCorr_allUnits_allEpochs[:] = np.nan
    rrCorr_medianUnits_allEpochs = np.empty(nb_epochs)
    rrCorr_medianUnits_allEpochs[:] = np.nan
    rrCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    rrCorr_allUnits_allEpochs[:] = np.nan
    

    obs_rate_allStimTrials = dataset_rr['stim_0']['val']
    num_iters = 10

    # check_trainVal_contamination(data_train.X,data_val.X,temporal_width)
    
    print('-----EVALUATING PERFORMANCE-----')
    for i in range(nb_epochs-1):
        try:
            weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
            mdl.load_weights(os.path.join(path_model_save,weight_file))
        except:
            weight_file = 'weights_'+fname_model+'_epoch-%03d' % (i+1)
            mdl.load_weights(os.path.join(path_model_save,weight_file))
        pred_rate = mdl.predict(data_val.X)
        _ = gc.collect()
        # val_loss,_,_,_ = mdl.evaluate(data_val.X,data_val.y,batch_size=data_val.X.shape[0])
        val_loss = None
        val_loss_allEpochs[i] = val_loss
        
        fev_loop = np.zeros((num_iters,n_cells))
        fracExVar_loop = np.zeros((num_iters,n_cells))
        predCorr_loop = np.zeros((num_iters,n_cells))
        rrCorr_loop = np.zeros((num_iters,n_cells))

        for j in range(num_iters):
            fev_loop[j,:], fracExVar_loop[j,:], predCorr_loop[j,:], rrCorr_loop[j,:] = model_evaluate_new(obs_rate_allStimTrials,pred_rate,temporal_width_eval,lag=samps_shift)
            
        fev = np.mean(fev_loop,axis=0)
        fracExVar = np.mean(fracExVar_loop,axis=0)
        predCorr = np.mean(predCorr_loop,axis=0)
        rrCorr = np.mean(rrCorr_loop,axis=0)


        # rgb = metrics.fraction_of_explainable_variance_explained(obs_rate,est_rate,unit_noise)
        fev_allUnits_allEpochs[i,:] = fev
        fev_medianUnits_allEpochs[i] = np.nanmedian(fev)      
        fracExVar_allUnits_allEpochs[i,:] = fracExVar
        fracExVar_medianUnits_allEpochs[i] = np.nanmedian(fracExVar)      
        
        predCorr_allUnits_allEpochs[i,:] = predCorr
        predCorr_medianUnits_allEpochs[i] = np.nanmedian(predCorr)
        rrCorr_allUnits_allEpochs[i,:] = rrCorr
        rrCorr_medianUnits_allEpochs[i] = np.nanmedian(rrCorr)
        

        _ = gc.collect()
    
    # fracExVar_allUnits = np.mean(fracExVar_allUnits_allEpochs,axis=0)
    # fracExVar_medianUnits = np.round(np.median(fracExVar_allUnits,axis=0),2)
    # rrCorr_allUnits = np.mean()
    # rrCorr_medianUnits = np.round(np.median(rrCorr_allUnits),2)

    
    idx_bestEpoch = np.nanargmax(fev_medianUnits_allEpochs)
    fev_medianUnits_bestEpoch = np.round(fev_medianUnits_allEpochs[idx_bestEpoch],2)
    fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
    fracExVar_medianUnits = np.round(fracExVar_medianUnits_allEpochs[idx_bestEpoch],2)
    fracExVar_allUnits = fracExVar_allUnits_allEpochs[(idx_bestEpoch),:]
    
    predCorr_medianUnits_bestEpoch = np.round(predCorr_medianUnits_allEpochs[idx_bestEpoch],2)
    predCorr_allUnits_bestEpoch = predCorr_allUnits_allEpochs[(idx_bestEpoch),:]
    rrCorr_medianUnits = np.round(rrCorr_medianUnits_allEpochs[idx_bestEpoch],2)
    rrCorr_allUnits = rrCorr_allUnits_allEpochs[(idx_bestEpoch),:]

    try:
        fname_bestWeight = 'weights_'+fname_model+'_epoch-%03d.h5' % (idx_bestEpoch+1)
        mdl.load_weights(os.path.join(path_model_save,fname_bestWeight))
    except:
        fname_bestWeight = 'weights_'+fname_model+'_epoch-%03d' % (idx_bestEpoch+1)
        mdl.load_weights(os.path.join(path_model_save,fname_bestWeight))
  
    pred_rate = mdl.predict(data_val.X)
    fname_bestWeight = np.array(fname_bestWeight,dtype='bytes')
    
    
 # %% Calculate new performance metrics and update the spreadsheets and model h5 files
    
    fname_save_performance = os.path.join(path_save_model_performance,(expDate+'_'+fname_model+'.h5'))

    print('-----SAVING PERFORMANCE STUFF TO H5-----')
    model_performance = {
        'fev_medianUnits_allEpochs': fev_medianUnits_allEpochs,
        'fev_allUnits_allEpochs': fev_allUnits_allEpochs,
        'fev_medianUnits_bestEpoch': fev_medianUnits_bestEpoch,
        'fev_allUnits_bestEpoch': fev_allUnits_bestEpoch,
        
        'fracExVar_medianUnits': fracExVar_medianUnits,
        'fracExVar_allUnits': fracExVar_allUnits,
        
        'predCorr_medianUnits_allEpochs': predCorr_medianUnits_allEpochs,
        'predCorr_allUnits_allEpochs': predCorr_allUnits_allEpochs,
        'predCorr_medianUnits_bestEpoch': predCorr_medianUnits_bestEpoch,
        'predCorr_allUnits_bestEpoch': predCorr_allUnits_bestEpoch,
        
        'rrCorr_medianUnits': rrCorr_medianUnits,
        'rrCorr_allUnits': rrCorr_allUnits,          
        
        'fname_bestWeight': np.atleast_1d(fname_bestWeight),
        'idx_bestEpoch': idx_bestEpoch,
        
        'val_loss_allEpochs': val_loss_allEpochs,
        'val_dataset_name': dataset_rr['stim_0']['dataset_name'],
        }
    
    if mdl_name[:2] == 'PR':
        weights = mdl.get_weights()
        model_performance['pr_alpha'] = weights[0]
        model_performance['pr_beta'] = weights[1]
        model_performance['pr_gamma'] = weights[2]
        model_performance['pr_tauY'] = weights[3]
        model_performance['pr_tauZ'] = weights[4]
        model_performance['pr_nY'] = weights[5]
        model_performance['pr_nZ'] = weights[6]


    metaInfo = {
       ' mdl_name': mdl.name,
        'path_model_save': path_model_save,
        'uname_selectedUnits': np.array(data_quality['uname_selectedUnits'],dtype='bytes'),#[idx_unitsToTake],dtype='bytes'),
        'idx_unitsToTake': idx_unitsToTake,
        'thresh_rr': thresh_rr,
        'trial_num': c_trial,
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
                'pr_temporal_width': pr_temporal_width
                }
    
    stim_info = {
         'fname_data_train_val_test':fname_data_train_val_test,
         'n_trainingSamps': data_train.X.shape[0],
         'n_valSamps': data_val.X.shape[0],
         'n_testSamps': data_test.X.shape[0],
         'temporal_width':temporal_width,
         'pr_temporal_width': pr_temporal_width
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
        'val_dataset_name': dataset_rr['stim_0']['dataset_name'],
        }

    dataset_rr = None
    save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred)

    
    
    
    # %% Write performance to csv file
    
    
    print('-----WRITING TO CSV FILE-----')
    if saveToCSV==1:
        csv_header = ['mdl_name','expDate','thresh_rr','RR','temp_window','batch_size','epochs','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','BatchNorm','MaxPool','c_trial','FEV_median','predCorr_median','rrCorr_median']
        csv_data = [mdl_name,expDate,thresh_rr,fracExVar_medianUnits,temporal_width,bz_ms,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,bn_val,mp_val,c_trial,fev_medianUnits_bestEpoch,predCorr_medianUnits_bestEpoch,rrCorr_medianUnits]
        
        fname_csv_file = fname_performance_excel
        if not os.path.exists(fname_csv_file):
            with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 
                
        with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_data) 

        
        
        
    print('-----FINISHED-----')

