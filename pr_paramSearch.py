#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:33:36 2021

@author: saad
"""

from model.RiekeModel import RiekeModel
from model.data_handler import load_h5Dataset, rolling_window, prepare_data_cnn2d, prepare_data_cnn3d
import numpy as np
import os
from collections import namedtuple 
Exptdata = namedtuple('Exptdata', ['X', 'y'])
import multiprocessing as mp
from joblib import Parallel, delayed
import time
import gc
import h5py
from model.performance import getModelParams, get_weightsDict, get_weightsOfLayer
from model.utils_si import splitall
from scipy.special import gamma as scipy_gamma
from scipy.signal import lfilter
from scipy import integrate
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Reshape

from model.load_savedModel import load
from model.performance import model_evaluate_new
import csv
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])


from tensorflow.keras.models import Model
import gc

import multiprocessing.process
multiprocessing.process.ORIGINAL_DIR = os.path.abspath(os.getcwd())

def run_model(pr_mdl_name,stim,resp,params,meanIntensity,upSampFac,downSampFac=17,n_discard=0,NORM=1,DOWN_SAMP=1,ROLLING_FAC=30,runOnGPU=False,ode_solver='RungeKutta',changeIntensities=False):
    
    stim_spatialDims = stim.shape[1:]
    stim = stim.reshape(stim.shape[0],stim.shape[1]*stim.shape[2])
    
    stim = np.repeat(stim,upSampFac,axis=0)
    
    if changeIntensities==True:
        stim[stim>0] = 2*meanIntensity
        stim[stim<0] = (2*meanIntensity)/300
        stim_photons = stim * params['timeStep']
        
    else:
        stim_photons = stim / upSampFac

    idx_allPixels = np.arange(0,stim.shape[1])
    
    # num_cores = mp.cpu_count()
    

    
    t = time.time()
    if pr_mdl_name == 'rieke':
        params['tme'] = np.arange(0,stim_photons.shape[0])*params['timeStep']
        params['biophysFlag'] = 1
        
        
        _,stim_currents = RiekeModel(params,stim_photons,ode_solver)
        
        
        if runOnGPU==True:
            stim_photons_tf = tf.convert_to_tensor(stim_photons,dtype=tf.float32)
            from model.RiekeModel import RiekeModel_tf
            _,stim_currents_tf = RiekeModel_tf(params,stim_photons_tf)
            
            # plt.plot(stim_currents_tf[:,0])
            # stim_currents_tf[0,0]
            stim_currents = np.array(stim_currents_tf)
        

        
    elif pr_mdl_name == 'clark':
        _,stim_currents = DA_model_iter(params,stim_photons)
        
        
    t_elasped_parallel = time.time()-t
    # print('time elasped: '+str(t_elasped_parallel)+' seconds')
    

    # reshape back to spatial pixels and downsample

    if DOWN_SAMP == 1 and downSampFac>1:
        rgb = stim_currents.T  
        rgb[np.isnan(rgb)] = np.nanmedian(rgb)
        
        rollingFac = ROLLING_FAC
        a = np.empty((len(idx_allPixels),rollingFac))
        a[:] = np.nan
        a = np.concatenate((a,rgb),axis=1)
        rgb8 = np.nanmean(rolling_window(a,rollingFac,time_axis = -1),axis=-1)
        rgb8 = rgb8.reshape(rgb8.shape[0],-1, downSampFac)    
        rgb8 = rgb8[:,:,0]
        
        
        stim_currents_downsampled = rgb8
        stim_currents_downsampled = stim_currents_downsampled.T
        


        
    else:
        stim_currents_downsampled = stim_currents
    
    stim_currents_reshaped = stim_currents_downsampled.reshape(stim_currents_downsampled.shape[0],stim_spatialDims[0],stim_spatialDims[1])
    stim_currents_reshaped = stim_currents_reshaped[n_discard:]
    
    if NORM==1:
        stim_currents_norm = (stim_currents_reshaped - np.min(stim_currents_reshaped)) / (np.max(stim_currents_reshaped)-np.min(stim_currents_reshaped))
        stim_currents_norm = stim_currents_norm - np.mean(stim_currents_norm)
    else:
        stim_currents_norm = stim_currents_reshaped
    
    # discard response if n_discard > 0
    if n_discard > 0:
        resp = resp[n_discard:]
        
    return stim_currents_norm,resp

def parser_pr_paramSearch():
    
    import argparse
    from model.parser import str2int


    parser = argparse.ArgumentParser()
    
    parser.add_argument('expDate',type=str)   
    parser.add_argument('path_mdl',type=str)
    parser.add_argument('trainingDataset',type=str)
    parser.add_argument('testingDataset',type=str)
    parser.add_argument('path_excel',type=str)
    parser.add_argument('path_perFiles',type=str)
    parser.add_argument('lightLevel',type=str)
    parser.add_argument('pr_type',type=str)
    parser.add_argument('pr_mdl_name',type=str)
    
    parser.add_argument('--r_sigma',type=str2int,default=7.66)
    parser.add_argument('--r_phi',type=str2int,default=7.66)
    parser.add_argument('--r_eta',type=str2int,default=1.62)
    parser.add_argument('--r_k',type=str2int,default=0.01)
    parser.add_argument('--r_h',type=str2int,default=3)
    parser.add_argument('--r_beta',type=str2int,default=25)
    parser.add_argument('--r_hillcoef',type=str2int,default=4)
    parser.add_argument('--r_gamma',type=str2int,default=800)
    
    parser.add_argument('--c_beta',type=str2int,default=0.36)
    parser.add_argument('--c_gamma',type=str2int,default=0.448)
    parser.add_argument('--c_tau_y',type=str2int,default=4.48)
    parser.add_argument('--c_n_y',type=str2int,default=4.33)
    parser.add_argument('--c_tau_z',type=str2int,default=166)
    parser.add_argument('--c_n_z',type=str2int,default=1)
    
    parser.add_argument('--mdl_name',type=str)
    parser.add_argument('--samps_shift',type=str2int,default=4)
    parser.add_argument('--num_cores',type=str2int,default=1)

    args = parser.parse_args()
    
    return args


# %%
def run_pr_paramSearch(expDate,path_mdl,trainingDataset,testingDataset,path_excel,path_perFiles,lightLevel,pr_type,pr_mdl_name,r_sigma=7.66,r_phi=7.66,r_eta=1.62,r_k=0.01,r_h=3,r_beta=25,r_hillcoef=4,r_gamma=800,c_beta=0.36,c_gamma=0.448,c_tau_y=4.48,c_n_y=4.33,c_tau_z=166,c_n_z = 1,
                       mdl_name='CNN_2D',idx_bestEpoch=-1,idx_mdl_start=0,idx_mdl_end=-1,samps_shift=0,num_cores=1,saveToH5='False',NORM_FIXED=0,timeBin=8):

    # from tensorflow.compat.v1 import ConfigProto
    # from tensorflow.compat.v1 import InteractiveSession
    
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)

# %%    
    
    DEBUG_MODE = 1
    # expDate = 'retina1'
    # lightLevel = 'scotopic'  # ['scotopic','photopic']
    # pr_type = 'rods'   # ['rods','cones']

    # path_dataset = os.path.join(path_mdl_drive,'datasets')
    # fname_dataset = expDate+'_dataset_train_val_test_'+lightLevel+'.h5'
    fname_data_train_val_test = testingDataset#os.path.join(path_dataset,fname_dataset)

    # mdlFolder = 'U-0.00_T-120_C1-13-03-50_C2-26-02-10_C3-24-01-62_BN-1_MP-0_TR-01'
    saveToCSV = 1
    # num_cores = 28
    # timeBin = 4

    
    
    
    # %% pr parameters
    if pr_mdl_name == 'rieke':
        params_cones = {}
        params_cones['sigma'] =  r_sigma #22  # rhodopsin activity decay rate (1/sec) - default 22
        params_cones['phi'] =  r_phi #22     # phosphodiesterase activity decay rate (1/sec) - default 22
        params_cones['eta'] =  r_eta #2000  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
        params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
        params_cones['k'] =  r_k #0.01     # constant relating cGMP to current - default 0.02
        params_cones['h'] = r_h #3       # cooperativity for cGMP->current - default 3
        params_cones['cdark'] =  1  # dark calcium concentration - default 1
        params_cones['beta'] = r_beta #16 # 9	  # rate constant for calcium removal in 1/sec - default 9
        params_cones['betaSlow'] =  0	  
        params_cones['hillcoef'] =  r_hillcoef #4  	  # cooperativity for cyclase, hill coef - default 4
        params_cones['hillaffinity'] =  0.5   # hill affinity for cyclase - default 0.5
        params_cones['gamma'] =  r_gamma/timeBin #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
        params_cones['timeStep'] =  1e-3  # freds default is 1e-4
        params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2
        
        # rods - mice
        params_rods = {}
        params_rods['sigma'] = r_sigma #30 # 7.66  # rhodopsin activity decay rate (1/sec) - default 22
        params_rods['phi'] =  r_phi #10 #7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
        params_rods['eta'] = r_eta #2.2 #1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
        params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
        params_rods['k'] =  r_k     # constant relating cGMP to current - default 0.02
        params_rods['h'] =  r_h       # cooperativity for cGMP->current - default 3
        params_rods['cdark'] =  1  # dark calcium concentration - default 1
        params_rods['beta'] =  r_beta	  # rate constant for calcium removal in 1/sec - default 9
        params_rods['betaSlow'] =  0	  
        params_rods['hillcoef'] =  r_hillcoef  	  # cooperativity for cyclase, hill coef - default 4
        params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
        params_rods['gamma'] =  r_gamma/timeBin #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
        params_rods['timeStep'] =  1e-3 # freds default is 1e-3
        params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2
        
    else:
        params_cones = {}
        params_cones['alpha'] =  1  
        params_cones['beta'] =  c_beta 
        params_cones['gamma'] =  c_gamma
        params_cones['tau_y'] =  c_tau_y #22
        params_cones['n_y'] =  c_n_y
        params_cones['tau_z'] =  c_tau_z
        params_cones['n_z'] =  c_n_z
        params_cones['timeStep'] = 1e-3
        params_cones['tau_r'] = 0 #4.78

        params_rods = {}
        params_rods['alpha'] =  1  
        params_rods['beta'] =  c_beta 
        params_rods['gamma'] =  c_gamma
        params_rods['tau_y'] =  c_tau_y #22
        params_rods['n_y'] =  c_n_y
        params_rods['tau_z'] =  c_tau_z
        params_rods['n_z'] =  c_n_z
        params_rods['timeStep'] = 1e-3
        params_rods['tau_r'] = 0 #4.78

    
    # %% Single pr type
    # timeBin = 8
    frameTime = 8
    NORM = 0
    DOWN_SAMP = 1
    ROLLING_FAC = 2
    upSampFac = int(frameTime/timeBin) #1#8 #17
    downSampFac = upSampFac
    meanIntensity = 0
    ode_solver = 'Euler'
    changeIntensities = False


    if DOWN_SAMP==0:
        ROLLING_FAC = 0
    else:
        ROLLING_FAC = 2
        
    
    
    if pr_type == 'cones':
        params = params_cones
        params_cones['timeStep'] = 1e-3*(frameTime/upSampFac)

    elif pr_type == 'rods':
        params = params_rods
        params_rods['timeStep'] = 1e-3*(frameTime/upSampFac)

    
    
    if pr_mdl_name == 'rieke':
        dataset_name = lightLevel+'_mdl-'+pr_mdl_name+'_s-'+str(params['sigma'])+'_p-'+str(params['phi'])+'_e-'+str(params['eta'])+'_k-'+str(params['k'])+'_h-'+str(params['h'])+'_b-'+str(params['beta'])+'_hc-'+str(params['hillcoef'])+'_gd-'+str(params['gdark'])+'_g-'+str(params['gamma'])+'_preproc-'+pr_type+'_norm-'+str(NORM)+'_tb-'+str(timeBin)+'_'+ode_solver+'_RF-'+str(ROLLING_FAC)
    elif pr_mdl_name == 'clark':
        dataset_name = lightLevel+'-'+str(meanIntensity)+'_mdl-'+pr_mdl_name+'_b-'+str(params['beta'])+'_g-'+str(params['gamma'])+'_y-'+str(params['tau_y'])+'_z-'+str(params['tau_z'])+'_preproc-'+pr_type+'_norm-'+str(NORM)+'_rfac-'+str(ROLLING_FAC)

    
    
    if DEBUG_MODE==1:
        nsamps_end = 10000  #10000
    else:
        nsamps_end = -1
    
    data_train_orig,data_val_orig,data_test,data_quality,dataset_rr,parameters,resp_orig = load_h5Dataset(fname_data_train_val_test,nsamps_val=-1,nsamps_train=nsamps_end,LOAD_ALL_TR=True)

    frames_X_orig = data_train_orig.X
    
    
    
    # Training data
    
    stim_train,resp_train = run_model(pr_mdl_name,data_train_orig.X,data_train_orig.y,params,meanIntensity,upSampFac,downSampFac=downSampFac,n_discard=1000,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC,ode_solver=ode_solver,changeIntensities=changeIntensities)
    
    value_min = -113.211676833235 #tf.math.reduce_min(inputs)
    value_max = -81.1005869186533 #tf.math.reduce_max(inputs)
    R_mean =  0.43907212331830137 #tf.math.reduce_mean(R_norm)       

    if NORM==1:
        if NORM_FIXED==1:
            value_min = -113.211676833235 #tf.math.reduce_min(inputs)
            value_max = -81.1005869186533 #tf.math.reduce_max(inputs)
            stim_train_med =  0.43907212331830137 #tf.math.reduce_mean(R_norm)       
            
            stim_train_norm = (stim_train - value_min)/(value_max-value_min)
            stim_train_norm = stim_train_norm - stim_train_med
            
        else:
            value_min = np.min(stim_train)
            value_max = np.max(stim_train)
            stim_train_norm = (stim_train - value_min)/(value_max-value_min)
            stim_train_med = np.nanmean(stim_train_norm)
            stim_train_norm = stim_train_norm - stim_train_med
    else:
        stim_train_norm = stim_train
    
    
    # Validation data
    n_discard_val = 50
    stim_val,resp_val = run_model(pr_mdl_name,data_val_orig.X,data_val_orig.y,params,meanIntensity,upSampFac,downSampFac=downSampFac,n_discard=n_discard_val,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC,runOnGPU=False,ode_solver=ode_solver)
    if NORM==1:
        if NORM_FIXED!=1:
            
            value_min = np.min(stim_val)
            value_max = np.max(stim_val)
    
        stim_val_norm = (stim_val - value_min)/(value_max-value_min)
        # stim_val_med = np.nanmean(stim_val_norm)
        stim_val_norm = stim_val_norm - stim_train_med
        
    else:
        stim_val_norm = stim_val 
    
    # Update dataset
    data_train = Exptdata(stim_train_norm,resp_train)
    data_val = Exptdata(stim_val_norm,resp_val)
    dataset_rr['stim_0']['val'] = dataset_rr['stim_0']['val'][:,n_discard_val:,:]
    
    # Update parameters
    for j in params.keys():
        parameters[j] = params[j]
    parameters['nsamps_end'] = nsamps_end
    
    
    # plt.plot(stim_val[:,0,0])
    # plt.ylim(y_lim)
    
    # plt.plot(stim_train_norm[:,0,0])
    # plt.plot(stim_val_norm[:,0,0])
    
    
    # %% Load model
    
    f_full = splitall(path_mdl)
    f = f_full[-1]
    rgb = getModelParams(f)
    select_T = rgb['T']
    correctMedian = False
    samps_shift = int(samps_shift)#0+4
    
    # path_model = os.path.join(path_mdl)
    mdl = load(os.path.join(path_mdl,f))
    
    fname_bestWeight = 'weights_'+f+'_epoch-%03d' % (idx_bestEpoch+1)
    try:
        mdl.load_weights(os.path.join(path_mdl,fname_bestWeight))
    except:
        mdl.load_weights(os.path.join(path_mdl,fname_bestWeight+'.h5'))
    weights_dict = get_weightsDict(mdl)
    

    
    # if mdl_name[:6]=='CNN_2D':
    data_val_prepared = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
    
    filt_temporal_width = select_T
    obs_rate_allStimTrials_d2 = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]
    obs_rate = data_val_prepared.y
    
    
    if idx_mdl_start>0:
        x = Input(shape=data_val_prepared.X.shape[1:])
        y = x 

        for layer in mdl.layers[idx_mdl_start:]:
            y = layer(y)
            
        mdl_d2 = Model(x, y, name='subset_model')
    
    old_mdl = mdl
    mdl = mdl_d2
    
    pred_rate = mdl.predict(data_val_prepared.X)
    
    
    num_iters = 50
    fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
    predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    
    for i in range(num_iters):
        fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_d2,pred_rate,0,RR_ONLY=False,lag=samps_shift)
    
    
    fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
    fracExplainableVar = np.mean(fracExplainableVar,axis=1)
    predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
    rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)
    
    fev_d2_medianUnits = np.median(fev_d2_allUnits)
    predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits)
    print(fev_d2_medianUnits)


    # idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
    # idx_d1_valid = idx_allUnits
    # idx_d1_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
    # idx_d1_valid = idx_allUnits[idx_d1_valid]


    # idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d1_valid])
    # idx_units_sorted = idx_d1_valid[idx_units_sorted]
    # idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[1],idx_units_sorted[0]]
    # # idx_unitsToPred = [26,3,16,32]

    # t_start = 10
    # t_dur = obs_rate.shape[0]
    # t_end = t_dur-20
    # win_display = (t_start,t_start+t_dur)
    # font_size_ticks = 14

    # t_frame = 8
    # t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

    # col_mdl = ('r')
    # lineWidth_mdl = [2]
    # lim_y = (0,6)
    # fig,axs = plt.subplots(2,2,figsize=(25,10))
    # axs = np.ravel(axs)



    # for i in range(len(idx_unitsToPred)):
    #     l_base, = axs[i].plot(t_axis[t_start+samps_shift:t_end],obs_rate[t_start:t_end-samps_shift,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
    #     l_base.set_label('Actual')
    #     l, = axs[i].plot(t_axis[t_start+samps_shift:t_end],pred_rate[t_start+samps_shift:t_end,idx_unitsToPred[i]],'r',linewidth=lineWidth_mdl[0])
    #     l.set_label('Predicted: FEV = %.02f, Corr = %0.2f' %(fev_d2_allUnits[idx_unitsToPred[i]],predCorr_d2_allUnits[idx_unitsToPred[i]]))
        
    #     # axs[i].set_ylim(lim_y)
    #     axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    #     axs[i].set_ylabel('Normalized spike rate (spikes/second)',fontsize=font_size_ticks)
    #     axs[i].set_title('unit_id: %d' %idx_unitsToPred[i])
    #     axs[i].legend()
    #     plt.setp(axs[i].get_xticklabels(), fontsize=font_size_ticks)

    
    
    
    # %% seperate out performance for transient and sustained cells
    # uname_selectedUnits = data_quality['uname_selectedUnits']
    # idx_sust = [i for i,v in enumerate(uname_selectedUnits) if '_bs_' in v]
    # idx_trans = [i for i,v in enumerate(uname_selectedUnits) if '_bt_' in v]

    # fev_d2_sustUnits = fev_d2_allUnits[idx_sust]
    # fev_d2_transUnits = fev_d2_allUnits[idx_trans]
    # fev_d2_median_sustUnits = np.nanmedian(fev_d2_sustUnits)
    # fev_d2_median_transUnits = np.nanmedian(fev_d2_transUnits)
    
    
    # predCorr_d2_sustUnits = predCorr_d2_allUnits[idx_sust]
    # predCorr_d2_transUnits = predCorr_d2_allUnits[idx_trans]
    # predCorr_d2_median_sustUnits = np.nanmedian(predCorr_d2_sustUnits)
    # predCorr_d2_median_transUnits = np.nanmedian(predCorr_d2_transUnits)

    
    
    # %% Write performance to csv file


    # print('-----WRITING TO CSV FILE-----')
    if saveToCSV==1:
        if pr_mdl_name=='rieke':
            csv_header = ['pr_mdl','params','sigma','phi','eta','k','h','beta','hillcoef','gamma','FEV_median','predCorr_median']
            csv_data = [pr_mdl_name,dataset_name,params['sigma'],params['phi'],params['eta'],params['k'],params['h'],params['beta'],params['hillcoef'],params['gamma']*timeBin,fev_d2_medianUnits,predCorr_d2_medianUnits]

            # csv_header = ['pr_mdl','params','sigma','phi','eta','k','h','beta','hillcoef','gamma','FEV_median','predCorr_median','FEV_sust','FEV_trans','predCorr_sust','predCorr_trans']
            # csv_data = [pr_mdl_name,dataset_name,params['sigma'],params['phi'],params['eta'],params['k'],params['h'],params['beta'],params['hillcoef'],params['gamma'],fev_d2_medianUnits,predCorr_d2_medianUnits,fev_d2_median_sustUnits,fev_d2_median_transUnits,predCorr_d2_median_sustUnits,predCorr_d2_median_transUnits]
        else:
            csv_header = ['pr_mdl','params','alpha','beta','gamma','tau_y','n_y','tau_z','n_z','tau_r','FEV_median','predCorr_median','FEV_sust','FEV_trans','predCorr_sust','predCorr_trans']
            csv_data = [pr_mdl_name,dataset_name,params['alpha'],params['beta'],params['gamma'],params['tau_y'],params['n_y'],params['tau_z'],params['n_z'],params['tau_r'],fev_d2_medianUnits,predCorr_d2_medianUnits,fev_d2_median_sustUnits,fev_d2_median_transUnits,predCorr_d2_median_sustUnits,predCorr_d2_median_transUnits]

        
        fname_csv_file = 'pr_paramSearch_'+lightLevel+'_'+pr_type+'_'+pr_mdl_name+'.csv'
        fname_csv_file = os.path.join(path_excel,fname_csv_file)
        if not os.path.exists(fname_csv_file):
            with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 
                
        with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_data) 
    
    
    # %% writing to h5
    # print('-----WRITING TO H5 FILE-----')
    

    performance = {
        'fev_d2_allUnits': fev_d2_allUnits,
        'predCorr_d2_allUnits': predCorr_d2_allUnits,
        'fev_d2_medianUnits': fev_d2_medianUnits,
        'predCorr_d2_medianUnits': predCorr_d2_medianUnits,
        
        # 'fev_d2_sustUnits': fev_d2_sustUnits,
        # 'fev_d2_transUnits': fev_d2_transUnits,
        # 'fev_d2_median_sustUnits': fev_d2_median_sustUnits,
        # 'fev_d2_median_transUnits': fev_d2_median_transUnits,
        
        # 'predCorr_d2_sustUnits': predCorr_d2_sustUnits,
        # 'predCorr_d2_transUnits': predCorr_d2_transUnits,
        # 'predCorr_d2_median_sustUnits': predCorr_d2_median_sustUnits,
        # 'predCorr_d2_median_transUnits': predCorr_d2_median_transUnits,
        
        # 'idx_sust': np.array(idx_sust),
        # 'idx_trans': np.array(idx_trans),
        # 'uname_selectedUnits': np.array(uname_selectedUnits,dtype='bytes')
        }
    
    if 'tme' in params:
        del params['tme']
    
    if saveToH5==1:
        fname_h5 = dataset_name+'_evaluation.h5'
        file_h5 = os.path.join(path_perFiles,fname_h5)
        f = h5py.File(file_h5,'w')
        
        grp = f.create_group('/performance')
        keys = list(performance.keys())
        for i in range(len(performance)):
            if performance[keys[i]].dtype == 'O':
                grp.create_dataset(keys[i], data=performance[keys[i]],dtype='bytes')
            else:
                grp.create_dataset(keys[i], data=performance[keys[i]])
        
        grp = f.create_group('/params')
        keys = list(params.keys())
        for i in range(len(params)):
            grp.create_dataset(keys[i], data=params[keys[i]])
        
        f.close()
        
        # print('-----DONE-----')

        
    # print('-----JOB FINISHED-----')
    
    return performance, params
    
# %%    
if __name__ == "__main__":
    args = parser_pr_paramSearch()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_pr_paramSearch(**vars(args))

