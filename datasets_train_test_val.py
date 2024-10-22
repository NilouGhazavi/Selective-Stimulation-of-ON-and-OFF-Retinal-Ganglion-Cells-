#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:34:08 2021

@author: saad
"""

# Save training, testing and validation datasets to be read by jobs on cluster

import os
import h5py
import numpy as np
from model.data_handler import load_data, load_data_kr_allLightLevels, save_h5Dataset, check_trainVal_contamination
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])

whos_data = 'kiersten'
lightLevel = 'allLightLevels'     # ['scotopic', 'photopic','scotopic_photopic']
datasetsToLoad = ['scotopic']#['scotopic','photopic','scotopic_photopic']
convertToRstar = False


if whos_data == 'saad':
    expDate = '20180502_s3'     # ('20180502_s3', '20180919_s3','20181211a_s3', '20181211b_s3')
    path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_saad/',expDate,'datasets')
elif whos_data == 'kiersten':
    expDate = 'retina1'     # ('retina1', 'retina2','retina3')
    path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/',expDate,'datasets/8ms_sampShifted')
    
    meanIntensities = {
    'scotopic': 1,
    'photopic': 10000
    }


fname_dataFile = os.path.join(path_dataset,(expDate+'_dataset_CB_'+lightLevel+'.h5'))


t_frame = 8
filt_temporal_width = 0
idx_cells = None
thresh_rr = 0

if whos_data == 'saad':
    frac_val = 0.2
    frac_test = 0.05
    
elif whos_data == 'kiersten':
    frac_val = 0
    frac_test = 0.05  

def applyLightIntensities(meanIntensity,data):

    X = data.X
    rgb = X[0]
    rgb = np.unique(rgb)
    if np.any(rgb<0):
        X[X>0] = 2*meanIntensity
        X[X<0] = (2*meanIntensity)/300
        
        X = X * 1e-3 * t_frame  # photons per time bin 

        
    data = Exptdata(X,data.y)
    return data



for d in datasetsToLoad:

    if whos_data == 'saad':
        data_train,data_val,data_test,data_quality,dataset_rr = load_data(fname_dataFile,frac_val=frac_val,frac_test=frac_test,filt_temporal_width=filt_temporal_width,idx_cells=idx_cells,thresh_rr=thresh_rr)
    elif whos_data == 'kiersten':    
        
        if lightLevel=='allLightLevels':
            data_train,data_val,data_test,data_quality,dataset_rr,resp_orig = load_data_kr_allLightLevels(fname_dataFile,d,frac_val=frac_val,frac_test=frac_test,filt_temporal_width=filt_temporal_width,idx_cells_orig=idx_cells,thresh_rr=thresh_rr)
    
        
        else:
            if lightLevel=='scotopic':
                rgb = os.path.join(path_dataset,(expDate+'_dataset_train_val_test_photopic.h5'))
                f = h5py.File(rgb,'r')
                idx_cells = np.array(f['data_quality']['idx_unitsToTake'])
    
            data_train,data_val,data_test,data_quality,dataset_rr = load_data_kr(fname_dataFile,frac_val=frac_val,frac_test=frac_test,filt_temporal_width=filt_temporal_width,idx_cells_orig=idx_cells,thresh_rr=thresh_rr)
    
    if convertToRstar == False:
        fname_data_train_val_test = os.path.join(path_dataset,(expDate+'_dataset_train_val_test_'+d+'.h5'))
    else:
        meanIntensity = meanIntensities[d]
        fname_data_train_val_test = os.path.join(path_dataset,(expDate+'_dataset_train_val_test_'+d+'-'+str(meanIntensity)+'.h5'))
        data_train = applyLightIntensities(meanIntensity,data_train)
        data_val = applyLightIntensities(meanIntensity,data_val)
        data_test = applyLightIntensities(meanIntensity,data_test)

    
    # idx_discard = check_trainVal_contamination(data_train.X,data_val.X,0)
    # if idx_discard.size!=0:
    #     idx_toKeep = np.sort(np.setdiff1d(np.arange(0,data_train.X.shape[0]),idx_discard))
    #     rgb_x = data_train.X[idx_toKeep]
    #     rgb_y = data_train.y[idx_toKeep]
    #     data_train = Exptdata(rgb_x,rgb_y)
    #     check_trainVal_contamination(data_train.X,data_val.X,0)
        
    
    f = h5py.File(fname_dataFile,'r')
    samps_shift = np.array(f[d]['val']['spikeRate'].attrs['samps_shift'])
    parameters = {
    't_frame': t_frame,
    'filt_temporal_width': filt_temporal_width,
    'frac_val': frac_val,
    'frac_test':frac_test,
    'thresh_rr': thresh_rr,
    'samps_shift': samps_shift
    }

    
    # save_h5Dataset(fname_data_train_val_test,data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=resp_orig)
    
    
# %% Shift photopic data by 5 samples
rgb_X = data_train.X
rgb_y = data_train.y

samps_shift = -9    # delay the responses by 5 frames

if samps_shift>0:
    rgb_X_shifted = rgb_X[:-samps_shift]
    rgb_y_shifted = rgb_y[samps_shift:]
    sign='+'

else:
    rgb_X_shifted = rgb_X[abs(samps_shift):]
    rgb_y_shifted = rgb_y[:-abs(samps_shift)]
    sign = ''



X_new = np.concatenate((data_train.X,rgb_X_shifted),axis=0)
y_new = np.concatenate((data_train.y,rgb_y_shifted),axis=0)

data_train = Exptdata(X_new,y_new)

    
fname_data_train_val_test = os.path.join(path_dataset,(expDate+'_dataset_train_val_test_'+d+'_'+'shifted'+d+'_'+sign+str(samps_shift)+'.h5'))
save_h5Dataset(fname_data_train_val_test,data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=resp_orig)


    