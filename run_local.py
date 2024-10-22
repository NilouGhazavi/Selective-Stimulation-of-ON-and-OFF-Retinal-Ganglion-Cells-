#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:42:54 2021

@author: saad
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from model.performance import getModelParams
from run_models import run_model

data_pers = 'kiersten' #'kiersten'
expDate = 'monkey01'
subFold = '' 
dataset = 'scot-3-30-Rstar' #'scot-30-Rstar_mdl-rieke_s-7.07_p-7.07_e-2.53_k-0.01_h-3_b-25_hc-4_gd-15.5_g-50.0_preproc-rods_norm-0_tb-8_Euler_RF-2'

idx_units_retrain = np.array([27,28,29,34,35,36])
idx_units_train = np.setdiff1d(np.arange(0,37),idx_units_retrain)

idx_units_ON = np.arange(0,30)
idx_units_ON_retrain = np.array([24,25,26,27,28,29])
idx_units_ON_train = np.setdiff1d(idx_units_ON,idx_units_ON_retrain)

idx_unitsToTake = 0#idx_units_ON_train #[0] #idx_units_train

mdl_subFold = 'LayerNorm_eps0'
mdl_name = 'PRFR_CNN2D_RODS' #'CNN_2D_NORM' #'BP_CNN2D' #'PRFR_CNN2D_RODS'#' #'PR_CNN2D_fixed' #'PR_CNN2D'#'CNN_2D' BP_CNN2D_MULTIBP_PRFRTRAINABLEGAMMA
path_existing_mdl = '' #'/home/saad/data/analyses/data_kiersten/monkey01/ClosedLoopTest/scot-30-Rstar_mdl-rieke_s-7.07_p-7.07_e-2.53_k-0.01_h-3_b-25_hc-4_gd-15.5_g-50.0_preproc-rods_norm-0_tb-8_Euler_RF-2/BP_CNN2D/U-24.00_P-180_T-120_CB-01_C1-08-09_C2-16-07_C3-18-05_BN-1_MP-1_LR-0.0010_TRSAMPS-030_TR-01'
info = ''
idxStart_fixedLayers = 0#1
idxEnd_fixedLayers = -1#15   #29 dense; 28 BN+dense; 21 conv+dense; 15 second conv; 8 first conv
CONTINUE_TRAINING = 1

lr = 0.0001
lr_fac = 1# how much to divide the learning rate when training is resumed
use_lrscheduler=0
USE_CHUNKER=1
pr_temporal_width = 180
temporal_width=120
thresh_rr=0
chans_bp = 0
chan1_n=8
filt1_size=9
filt1_3rdDim=0
chan2_n=16
filt2_size=7
filt2_3rdDim=0
chan3_n=18
filt3_size=5
filt3_3rdDim=0
nb_epochs=70#42         # setting this to 0 only runs evaluation
bz_ms=5000#5000
BatchNorm=1
MaxPool=1
runOnCluster=0
num_trials=1

BatchNorm_train = 0
saveToCSV=1
trainingSamps_dur=40 # minutes
validationSamps_dur=0.2

name_datasetFile = expDate+'_dataset_train_val_test_'+dataset+'.h5'
path_model_save_base = os.path.join('/home/saad/data/analyses/data_'+data_pers+'/',expDate,subFold,dataset,mdl_subFold)
path_dataset_base = os.path.join('/home/saad/data/analyses/data_'+data_pers+'/',expDate,subFold)
fname_data_train_val_test = os.path.join(path_dataset_base,'datasets',name_datasetFile)

c_trial = 1

if path_existing_mdl=='' and idxStart_fixedLayers>0:
    raise ValueError('Transfer learning set. Define existing model path')

    
# %%
for c_trial in range(1,num_trials+1):
    model_performance,mdl = run_model(expDate,mdl_name,path_model_save_base,fname_data_train_val_test,path_dataset_base=path_dataset_base,saveToCSV=saveToCSV,runOnCluster=0,
                            temporal_width=temporal_width, pr_temporal_width=pr_temporal_width, thresh_rr=thresh_rr,
                            chans_bp=chans_bp,
                            chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim,
                            chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim,
                            chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim,
                            nb_epochs=nb_epochs,bz_ms=bz_ms,
                            BatchNorm=BatchNorm,BatchNorm_train = BatchNorm_train,MaxPool=MaxPool,c_trial=c_trial,USE_CHUNKER=USE_CHUNKER,
                            path_existing_mdl = path_existing_mdl, idxStart_fixedLayers=idxStart_fixedLayers, idxEnd_fixedLayers=idxEnd_fixedLayers,
                            CONTINUE_TRAINING=CONTINUE_TRAINING,info=info,
                            trainingSamps_dur=trainingSamps_dur,validationSamps_dur=validationSamps_dur,idx_unitsToTake=idx_unitsToTake,
                            lr=lr,lr_fac=lr_fac,use_lrscheduler=use_lrscheduler)
    
plt.plot(model_performance['fev_medianUnits_allEpochs'])
print('FEV = %0.2f' %(np.nanmax(model_performance['fev_medianUnits_allEpochs'])*100))



# %% Evaluate several models in loop
import numpy as np
import matplotlib.pyplot as plt
import os
from model.performance import getModelParams
from run_models import run_model
import h5py

min_nepochs = 30

data_pers = 'kiersten'
expDate = 'retina1'
subFold = 'PR_BP' #'8ms_clark'
dataset = 'photopic-10000_mdl-rieke_s-22_p-22_e-2000_k-0.01_h-3_b-9_hc-4_gd-28_preproc-cones_norm-1_tb-4_Euler_RF-2'

mdl_subFold = 'paramSearch'
mdl_name = 'BP_CNN2D_MULTIBP'#' #'PR_CNN2D_fixed' #'PR_CNN2D'#'CNN_2D' BP_CNN2D_MULTIBP_PRFRTRAINABLEGAMMA

name_datasetFile = expDate+'_dataset_train_val_test_'+dataset+'.h5'
path_model_save_base = os.path.join('/home/saad/data/analyses/data_'+data_pers+'/',expDate,subFold,dataset,mdl_subFold)
path_dataset_base = os.path.join('/home/saad/data/analyses/data_'+data_pers+'/',expDate,subFold)
fname_data_train_val_test = os.path.join(path_dataset_base,'datasets',name_datasetFile)

fname_allParams = os.listdir(os.path.join(path_model_save_base,mdl_name))

recalculate_perf = 1

counter = 0
for f in fname_allParams:
    counter = counter+1
    print('%d of %d\n' %(counter,len(fname_allParams)))
    current_epoch = len([f for f in os.listdir(os.path.join(path_model_save_base,mdl_name,f)) if f.endswith('index')])
    
    fname_performance = os.path.join(path_model_save_base,mdl_name,f,'performance',expDate+'_'+f+'.h5')
    perf_exist = os.path.exists(fname_performance)
    try:
        perf = h5py.File(fname_performance,'r')
        nepochsAtPerfCalc = perf['model_performance']['fev_medianUnits_allEpochs'].shape[0]
        perf.close()
    except:
        nepochsAtPerfCalc = 0
    
    
    # if current_epoch>min_nepochs and not(recalculate_perf==0 and perf_exist):
    if current_epoch>min_nepochs and nepochsAtPerfCalc<min_nepochs:
            
        params = getModelParams(f)
        
    
        lr = params['LR']
        pr_temporal_width = params['P']
        temporal_width=params['T']
        thresh_rr=params['U']
        chan1_n=params['C1_n']
        filt1_size=params['C1_s']
        filt1_3rdDim=params['C1_3d']
        chan2_n=params['C2_n']
        filt2_size=params['C2_s']
        filt2_3rdDim=params['C2_3d']
        chan3_n=params['C3_n']
        filt3_size=params['C3_s']
        filt3_3rdDim=params['C3_3d']
        trainingSamps_dur = 0 # minutes
        nb_epochs=0         # setting this to 0 only runs evaluation
        bz_ms=400#20000 #10000
        USE_CHUNKER = 1
        BatchNorm=params['BN']
        MaxPool=params['MP']
        runOnCluster=0
        c_trial = params['TR']
        
        BatchNorm_train = 0
        saveToCSV=1
        trainingSamps_dur=0
        validationSamps_dur=0
        

    
        
        model_performance,mdl = run_model(expDate,mdl_name,path_model_save_base,fname_data_train_val_test,path_dataset_base=path_dataset_base,saveToCSV=saveToCSV,runOnCluster=0,
                                temporal_width=temporal_width, pr_temporal_width=pr_temporal_width, thresh_rr=thresh_rr,
                                chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim,
                                chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim,
                                chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim,
                                nb_epochs=nb_epochs,bz_ms=bz_ms,
                                BatchNorm=BatchNorm,BatchNorm_train = BatchNorm_train,MaxPool=MaxPool,c_trial=c_trial,CONTINUE_TRAINING=1,info='',
                                trainingSamps_dur=trainingSamps_dur,validationSamps_dur=validationSamps_dur,lr=lr,USE_CHUNKER=USE_CHUNKER)
        
        # plt.plot(model_performance['fev_medianUnits_allEpochs'])
        # plt.title(f)
        # print('Model: %s\nFEV = %0.2f\n' %(f,(np.max(model_performance['fev_medianUnits_allEpochs'])*100)))
    
    
X.sha