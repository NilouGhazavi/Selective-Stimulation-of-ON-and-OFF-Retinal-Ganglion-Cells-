#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:34:04 2021

@author: saad
"""

import numpy as np
import csv
import os

missingFnames = set(fname_model)-set(paramNames_allExps['retina1']['CNN_2D'])

missingParams = dict([(key, []) for key in param_list_keys])
for i in missingFnames:
    rgb = getModelParams(i)
    for i in param_list_keys:
        missingParams[i].append(rgb[i])


expDate = 'retina1'
path_model_save_base = os.path.join('/home/sidrees/scratch/RetinaPredictors/data')
mdl_name = 'CNN_2D'
thresh_rr=0.15
temporal_width=60
bz_ms=10000
nb_epochs=150

BatchNorm=1
MaxPool=0
num_trials=1


fname_csv_file = 'model_params.csv'
if os.path.exists(fname_csv_file):
    raise ValueError('Paramter file already exists')

csv_header = ['expDate','mdl_name','path_model_save_base','thresh_rr','temp_width','bz_ms','nc_epochs','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','BatchNorm','MaxPool','num_trials']
params_array = np.zeros((10000,3*3))
counter = -1

fname_model_2 = ([])
for i in range(len(missingFnames)):
    csv_data = [expDate,mdl_name,path_model_save_base,thresh_rr,temporal_width,bz_ms,nb_epochs,
                missingParams['C1_n'][i],missingParams['C1_s'][i],missingParams['C1_3d'][i],
                missingParams['C2_n'][i],missingParams['C2_s'][i],missingParams['C2_3d'][i],
                missingParams['C3_n'][i],missingParams['C3_s'][i],missingParams['C3_3d'][i],
                missingParams['BN'][i],missingParams['MP'][i],missingParams['TR'][i]]
    
    fname_model_2.append('U-%0.2f_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(csv_data[3],csv_data[4],csv_data[7],
                                                                                                                    csv_data[8],csv_data[10],csv_data[11], 
                                                                                                                    csv_data[13],csv_data[14], 
                                                                                                                    csv_data[16],csv_data[17],csv_data[18]))
    
    # if not os.path.exists(fname_csv_file):
    #     with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
    #         csvwriter = csv.writer(csvfile) 
    #         csvwriter.writerow(csv_header) 
            
    # with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
    #     csvwriter = csv.writer(csvfile) 
    #     csvwriter.writerow(csv_data) 

