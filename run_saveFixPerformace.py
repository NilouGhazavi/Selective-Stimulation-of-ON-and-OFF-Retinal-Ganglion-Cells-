#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:42:54 2021

@author: saad
"""


import numpy as np
import os

from fix_savePerformance import run_fixPerformance
from model.performance import getModelParams

expDate = 'retina1'
samps_shift = 0
subFold = '8ms_trainablePR'
dataset_subFold = 'largeGamma'
dataset = 'scotopic-1' #'photopic-10000_preproc-added_norm-1_rfac-2'
mdl_name = 'PRFR_CNN2D_fixed'
path_existing_mdl = '/home/saad/data/analyses/data_kiersten/retina3/8ms_resamp/photopic-10000_mdl-rieke_s-250_p-40.7_e-879_k-0.01_h-3_b-110_hc-2.64_gd-28_preproc-cones_norm-1_tb-4_RungeKutta/CNN_2D/U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_TR-01'
idx_CNN_start = 1
temporal_width=120
thresh_rr=0
chan1_n=13
filt1_size=3
filt1_3rdDim=0
chan2_n=26
filt2_size=2
filt2_3rdDim=0
chan3_n=24
filt3_size=1
filt3_3rdDim=0
# nb_epochs=20
bz_ms=20000
BatchNorm=0
MaxPool=0
saveToCSV=1
runOnCluster=0
num_trials=1
c_trial = 1

name_datasetFile = expDate+'_dataset_train_val_test_'+dataset+'.h5'
path_model_save_base = os.path.join('/home/saad/data/analyses/data_kiersten/',expDate,subFold,dataset,dataset_subFold)
path_dataset_base = os.path.join('/home/saad/data/analyses/data_kiersten/',expDate,subFold)

param_list_keys = ['U','P', 'T','C1_n','C1_s','C1_3d','C2_n','C2_s','C2_3d','C3_n','C3_s','C3_3d','BN','MP','TR']
params = dict([(key, []) for key in param_list_keys])
paramFileNames = os.listdir(os.path.join(path_model_save_base,mdl_name))
for f in paramFileNames:
    rgb = getModelParams(f)
    for i in rgb.keys():
        params[i].append(rgb[i])

rangeToRun = np.arange(0,len(paramFileNames))
fname_performance_excel = os.path.join('/home/saad/postdoc_db/projects/RetinaPredictors/performance/','performance_'+expDate+'_'+dataset+'_'+str(rangeToRun[0])+'-'+str(rangeToRun[-1])+'.csv')

i = 0
temporal_width = params['T'][i]
pr_temporal_width = params['P'][i]
chan1_n=params['C1_n'][i]
filt1_size=params['C1_s'][i]
filt1_3rdDim=params['C1_3d'][i]
chan2_n=params['C2_n'][i]
filt2_size=params['C2_s'][i]
filt2_3rdDim=params['C2_3d'][i]
chan3_n=params['C3_n'][i]
filt3_size=params['C3_s'][i]
filt3_3rdDim=params['C3_3d'][i]
# nb_epochs=nb_epochs
bz_ms=bz_ms
BatchNorm=params['BN'][i]
MaxPool=MaxPool
c_trial=params['TR'][i]

#%%
for i in rangeToRun[0:]:
    prog = '%d of %d' %(i,rangeToRun[-1])
    print(prog)
    if os.path.exists(os.path.join(path_model_save_base,mdl_name,paramFileNames[i],paramFileNames[i])):
        model_performance = run_fixPerformance(expDate,mdl_name,path_model_save_base,name_datasetFile,path_dataset_base=path_dataset_base,path_existing_mdl=path_existing_mdl,
                            fname_performance_excel=fname_performance_excel,saveToCSV=saveToCSV,runOnCluster=0,
                            temporal_width=temporal_width, pr_temporal_width = pr_temporal_width, thresh_rr=thresh_rr,samps_shift=samps_shift,
                            chan1_n=params['C1_n'][i], filt1_size=params['C1_s'][i], filt1_3rdDim=params['C1_3d'][i],
                            chan2_n=params['C2_n'][i], filt2_size=params['C2_s'][i], filt2_3rdDim=params['C2_3d'][i],
                            chan3_n=params['C3_n'][i], filt3_size=params['C3_s'][i], filt3_3rdDim=params['C3_3d'][i],
                            bz_ms=bz_ms,BatchNorm=params['BN'][i],MaxPool=MaxPool,c_trial=params['TR'][i],idx_CNN_start=idx_CNN_start)


#%%

    
    
    
    # model_performance = run_fixPerofrmance(expDate,mdl_name,path_model_save_base,path_dataset_base=path_dataset_base,saveToCSV=saveToCSV,runOnCluster=0,
    #                     temporal_width=temporal_width, thresh_rr=thresh_rr,
    #                     chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim,
    #                     chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim,
    #                     chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim,
    #                     nb_epochs=nb_epochs,bz_ms=bz_ms,BatchNorm=BatchNorm,MaxPool=MaxPool,c_trial=c_trial)
