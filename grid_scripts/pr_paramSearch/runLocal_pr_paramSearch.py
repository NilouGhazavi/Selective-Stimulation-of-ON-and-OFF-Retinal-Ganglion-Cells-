#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:38:04 2021

@author: saad
"""
from pr_paramSearch import run_pr_paramSearch
import numpy as np

expDate = 'monkey01'
samps_shift = 0
lightLevel = 'scot-0.3-3-Rstar'
pr_type = 'rods'
pr_mdl_name = 'rieke'
path_mdl = '/home/saad/data/analyses/data_kiersten/monkey01/scot-0.3-3-Rstar/PRFR_CNN2D_RODS/U-0.00_P-180_T-120_C1-08-09_C2-16-07_C3-18-05_BN-1_MP-1_LR-0.0010_TR-01'
trainingDataset = ''
testingDataset = '/home/saad/data/analyses/data_kiersten/'+expDate+'/datasets/'+expDate+'_dataset_train_val_test_'+lightLevel+'.h5'
mdl_name = 'PRFR_CNN2D_RODS'
idx_bestEpoch = 75 #22
idx_mdl_start = 5
idx_mdl_end = -1

path_excel = '/home/saad/data/analyses/data_kiersten/'+expDate+'/pr_diff'
path_perFiles = '/home/saad/data/analyses/data_kiersten/'+expDate+'/pr_diff'



timeBin = 8
r_sigma_arr = np.arange(0,30,5)[1:] #15.6
r_phi_arr = np.arange(0,30,5)[1:]
r_eta_arr = np.arange(10,100,10)
r_k_arr = np.arange(0.01,0.02)
r_h_arr = np.arange(3,4)
r_beta_arr = np.arange(10,50,10)
r_hillcoef_arr = np.arange(4,5) 
r_gamma_arr = np.arange(0,100,20)[1:]

params_array = np.zeros((100000,8))
cntr = -1
for s in r_sigma_arr:
    for p in r_phi_arr:
        for e in r_eta_arr:
            for k in r_k_arr:
                for h in r_h_arr:
                    for b in r_beta_arr:
                        for hc in r_hillcoef_arr:
                            for g in r_gamma_arr:
                                cntr+=1
                                params_array[cntr] = [s,p,e,k,h,b,hc,g]
                                
params_array = params_array[:cntr+1]
params_array.shape

i = 0
for i in range(params_array.shape[0]):
    print('%d of %d' %(i,params_array.shape[0]))
    r_sigma = params_array[i,0]
    r_phi = params_array[i,1]
    r_eta = params_array[i,2]
    r_k = params_array[i,3]
    r_h = params_array[i,4]
    r_beta = params_array[i,5]
    r_hillcoef = params_array[i,6]
    r_gamma = params_array[i,7]
    
    perf,_ = run_pr_paramSearch(expDate,path_mdl,trainingDataset,testingDataset,path_excel,path_perFiles,lightLevel,pr_type,pr_mdl_name,samps_shift=samps_shift,
                                r_sigma=r_sigma,r_phi=r_phi,r_eta=r_eta,r_k=r_k,r_h=r_h,r_beta=r_beta,r_hillcoef=r_hillcoef,r_gamma=r_gamma,
                                mdl_name=mdl_name,idx_bestEpoch=idx_bestEpoch,idx_mdl_start=idx_mdl_start,idx_mdl_end=idx_mdl_end,num_cores=28,timeBin=timeBin)
    print('FEV = %0.2f' %(np.nanmax(perf['fev_d2_medianUnits'])*100))

# run_pr_paramSearch(expDate,path_mdl,trainingDataset,testingDataset,path_excel,path_perFiles,lightLevel,pr_type,pr_mdl_name,samps_shift=samps_shift,c_beta=c_beta,c_gamma=c_gamma,c_tau_y=c_tau_y,c_n_y=c_n_y,c_tau_z=c_tau_z,c_n_z=c_n_z,mdl_name=mdl_name,num_cores=28)