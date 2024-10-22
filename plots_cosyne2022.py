#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:01:49 2021

@author: saad
"""
import h5py
import numpy as np
import os
import re

from global_scripts import utils_si
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pylustrator
# pylustrator.start()


from scipy.stats import wilcoxon
import gc
 
import csv
import pylustrator
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])

from model.load_savedModel import load
from model.data_handler import load_data, load_h5Dataset, prepare_data_cnn2d, prepare_data_cnn3d, prepare_data_convLSTM, prepare_data_pr_cnn2d
from model.performance import getModelParams, model_evaluate,model_evaluate_new,paramsToName, get_weightsDict
from model import metrics
from model import featureMaps
from pyret.filtertools import sta, decompose

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    
   
# %% a. example fig - with PR
exp_select = 'retina1'
path_exp = '/home/saad/data/analyses/data_kiersten/'+exp_select+'/8ms_resamp'
mdl_params = 'U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_TR-01'

mdlFolder = os.path.join(path_exp,'photopic-10000_mdl-rieke_s-250_p-40.7_e-879_k-0.01_h-3_b-110_hc-2.64_gd-28_preproc-cones_norm-1_tb-4_RungeKutta_RF-2/CNN_2D/',mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


val_dataset_2 = 'scotopic-1_mdl-rieke_s-10_p-10_e-20_g-2.5_k-0.01_h-3_b-15.7_hc-5.2_gd-20_preproc-added_norm-1_tb-4_RungeKutta_RF-2' 
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0


obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)
obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]

num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)*100
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
idx_d2_valid = idx_allUnits
idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<100)
idx_d2_valid = idx_allUnits[idx_d2_valid]

fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)

predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)

idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
idx_units_sorted = idx_d2_valid[idx_units_sorted]
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[-3],idx_units_sorted[-4]]
idx_unitsToPred = [32]

t_start = 0
t_dur = obs_rate.shape[0]
t_end = t_dur-20
win_display = (t_start,t_start+t_dur)

t_frame = 8
t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

col_mdl = ('r')
lim_y = (-0.2,7)

# fig,axs = plt.subplots(1,1,figsize=(25,10))
font_size_ticks = 9
font_size_labels = 9
font_size_legend = 8
font_size_title = 9

pylustrator.start()
figure_1a,axs = plt.subplots(1,1)
axs = np.ravel(axs)
figure_1a.suptitle('')

for i in range(1):
    l_base, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate[t_start:t_end-samps_shift_2,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate[t_start+samps_shift_2:t_end,idx_unitsToPred[i]],color='darkgreen',linewidth=3)
    l.set_label('Predicted:\nFEV = %02d%%\nCorr = %0.2f' %(fev_d2_allUnits[idx_unitsToPred[i]],predCorr_d2_allUnits[idx_unitsToPred[i]]))
    
    axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
    axs[i].set_title('Example RGC (unit-'+str(idx_unitsToPred[0])+'): Model with photoreceptor layer',fontsize=font_size_title)
    axs[i].text(0.75,6,'Trained at photopic light level\nEvaluated at scotopic light level',fontsize=font_size_legend)
    axs[i].legend(loc='upper right',fontsize=font_size_legend)
    plt.setp(axs[i].get_xticklabels(), fontsize=font_size_ticks)
    plt.setp(axs[i].get_yticklabels(), fontsize=font_size_ticks)

# % start: automatic generated code from pylustrator
# plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
# import matplotlib as mpl
# plt.figure(1).set_size_inches(9.017000/2.54, 4.953000/2.54, forward=True)
# plt.figure(1).axes[0].set_position([0.111895, 0.128205, 0.868395, 0.774359])
# plt.figure(1).axes[0].title.set_fontsize(8)
# plt.figure(1).axes[0].title.set_ha("center")
# plt.figure(1).axes[0].title.set_position([0.478457, 0.995913])
# plt.figure(1).axes[0].title.set_weight("normal")
# plt.figure(1).axes[0].xaxis.labelpad = -3.997258
# plt.figure(1).axes[0].xaxis.labelpad = -3.922742
# plt.figure(1).axes[0].yaxis.labelpad = 0.320000
# plt.figure(1).axes[0].texts[0].set_position([0.750000, 5.908709])
# plt.figure(1).texts[0].set_position([0.532086, 1.005128])
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
# plt.figure(1).texts[1].set_fontsize(8)
# plt.figure(1).texts[1].set_position([0.575764, 0.946203])
# plt.figure(1).texts[1].set_text("with")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
# plt.figure(1).texts[2].set_fontsize(8)
# plt.figure(1).texts[2].set_position([0.575764, 0.946203])
# plt.figure(1).texts[2].set_text("with")
# #% end: automatic generated code from pylustrator
# #% start: automatic generated code from pylustrator
# plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
# import matplotlib as mpl
# plt.figure(1).set_size_inches(9.144000/2.54, 5.080000/2.54, forward=True)
# plt.figure(1).axes[0].set_position([0.135473, 0.175646, 0.844213, 0.717595])
# plt.figure(1).axes[0].title.set_fontsize(9)
# plt.figure(1).axes[0].title.set_position([0.430170, 1.002795])
# plt.figure(1).axes[0].xaxis.labelpad = -0.006027
# plt.figure(1).axes[0].yaxis.labelpad = 2.873311
# plt.figure(1).axes[0].get_xaxis().get_label().set_fontsize(9)
# plt.figure(1).axes[0].get_yaxis().get_label().set_fontsize(9)
# plt.figure(1).texts[0].set_position([0.543963, 0.988287])
# plt.figure(1).texts[1].set_fontsize(9)
# plt.figure(1).texts[1].set_position([0.553126, 0.936205])
# plt.figure(1).texts[2].set_position([0.586425, 0.933682])
# plt.figure(1).texts[2].set_visible(False)
#% end: automatic generated code from pylustrator

# plt.show()
# plt.savefig("sample.jpg", dpi=600)

# %% b. example fig - without PR
exp_select = 'retina1'
path_exp = '/home/saad/data/analyses/data_kiersten/'+exp_select+'/8ms'
mdl_params = 'U-0.00_T-120_C1-20-03_C2-24-02_C3-22-01_BN-1_MP-0_TR-01'

mdlFolder = os.path.join(path_exp,'photopic/CNN_2D/',mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


val_dataset_2 = 'scotopic' 
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0

idx_start = 50
obs_rate = data_val.y[idx_start:]
pred_rate = mdl.predict(data_val.X[idx_start:])
obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]
obs_rate_allStimTrials_scotpic = obs_rate_allStimTrials_scotpic[:,idx_start:,:]

num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)*100
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
idx_d2_valid = idx_allUnits
# idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<100)
# idx_d2_valid = idx_allUnits[idx_d2_valid]

fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)

predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)

idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
idx_units_sorted = idx_d2_valid[idx_units_sorted]
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[-3],idx_units_sorted[-4]]
idx_unitsToPred = [32]

t_start = 0 #10 + 45
t_dur = obs_rate.shape[0]
t_end = t_dur-20
win_display = (t_start,t_start+t_dur)

t_frame = 8
t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

col_mdl = ('r')
lim_y = (-0.2,7)


font_size_ticks = 9
font_size_labels = 9
font_size_legend = 8
font_size_title = 9

# %%
pylustrator.start()
figure,axs = plt.subplots(1,1)
axs = np.ravel(axs)
figure.suptitle('')

for i in range(1):
    l_base, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate[t_start:t_end-samps_shift_2,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate[t_start+samps_shift_2:t_end,idx_unitsToPred[i]],color='blue',linewidth=3)
    l.set_label('Predicted:\nFEV = %02d%%\nCorr = %0.2f' %(fev_d2_allUnits[idx_unitsToPred[i]],predCorr_d2_allUnits[idx_unitsToPred[i]]))
    
    axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
    axs[i].set_title('Example RGC (unit-'+str(idx_unitsToPred[0])+'): Model without photoreceptor layer',fontsize=font_size_title)
    axs[i].text(0.75,6,'Trained at photopic light level\nEvaluated at scotopic light level',fontsize=font_size_legend)
    plt.setp(axs[i].get_xticklabels(), fontsize=font_size_labels)
    plt.setp(axs[i].get_yticklabels(), fontsize=font_size_labels)
    axs[i].legend(loc='upper right',fontsize=font_size_legend)


#% start: automatic generated code from pylustrator
# plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
# import matplotlib as mpl
# plt.figure(1).set_size_inches(9.448800/2.54, 5.334000/2.54, forward=True)
# plt.figure(1).axes[0].set_position([0.117855, 0.169104, 0.862251, 0.727088])
# plt.figure(1).axes[0].title.set_fontsize(9)
# plt.figure(1).axes[0].title.set_ha("center")
# plt.figure(1).axes[0].title.set_position([0.430170, 1.000000])
# plt.figure(1).axes[0].title.set_weight("normal")
# plt.figure(1).axes[0].xaxis.labelpad = -0.991135
# plt.figure(1).axes[0].yaxis.labelpad = 0.320000
# plt.figure(1).axes[0].yaxis.labelpad = -1.153324
# plt.figure(1).axes[0].texts[0].set_position([0.750000, 5.908709])
# plt.figure(1).axes[0].title.set_position([0.440198, 1.000000])
# plt.figure(1).texts[0].set_position([0.503214, 0.968901])
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
# plt.figure(1).texts[1].set_fontsize(9)
# plt.figure(1).texts[1].set_position([0.521948, 0.937198])
# plt.figure(1).texts[1].set_text("without")
# #% end: automatic generated code from pylustrator
# plt.savefig("sample.jpg", dpi=600)
# plt.show()


# %% c. scotopic with and without PR - BAR PLOTS

exp_select = 'retina2'
path_exp = '/home/saad/data/analyses/data_kiersten/'+exp_select+'/8ms_resamp'
mdl_params = 'U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_TR-01'

mdlFolder = os.path.join(path_exp,'photopic-10000_mdl-rieke_s-250_p-40.7_e-879_k-0.01_h-3_b-110_hc-2.64_gd-28_preproc-cones_norm-1_tb-4_RungeKutta_RF-2/CNN_2D/',mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


val_dataset_2 = 'scotopic-1_mdl-rieke_s-10_p-10_e-20_g-2.5_k-0.01_h-3_b-15.7_hc-5.2_gd-20_preproc-added_norm-1_tb-4_RungeKutta_RF-2' 
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0


obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)
obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]

num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
idx_d2_valid = idx_allUnits
idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
idx_d2_valid = idx_allUnits[idx_d2_valid]

m1_fev_d2_allUnits = fev_d2_allUnits*100
m1_fev_d2_medianUnits = np.median(m1_fev_d2_allUnits[idx_d2_valid])
m1_fev_d2_stdUnits = np.std(m1_fev_d2_allUnits[idx_d2_valid])
m1_fev_d2_ci = 1.96*(m1_fev_d2_stdUnits/len(idx_d2_valid)**.5)

m1_predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
m1_predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
m1_predCorr_d2_ci = 1.96*(m1_predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

# ------- WITHOUT PR ------- #
path_exp = '/home/saad/data/analyses/data_kiersten/'+exp_select+'/8ms'
mdl_params = 'U-0.00_T-120_C1-20-03_C2-24-02_C3-22-01_BN-1_MP-0_TR-01'


mdlFolder = os.path.join(path_exp,'photopic/CNN_2D/',mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d.h5' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


val_dataset_2 = 'scotopic' 
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0


obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)
obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]

num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
idx_d2_valid = idx_allUnits
idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
idx_d2_valid = idx_allUnits[idx_d2_valid]

m2_fev_d2_allUnits = fev_d2_allUnits*100
m2_fev_d2_medianUnits = np.median(m2_fev_d2_allUnits[idx_d2_valid])
m2_fev_d2_stdUnits = np.std(m2_fev_d2_allUnits[idx_d2_valid])
m2_fev_d2_ci = 1.96*(m2_fev_d2_stdUnits/len(idx_d2_valid)**.5)

m2_predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
m2_predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
m2_predCorr_d2_ci = 1.96*(m2_predCorr_d2_stdUnits/len(idx_d2_valid)**.5)


# % ------------ Bar plots  --------------- #

font_size_ticks = 8
font_size_labels = 8
font_size_legend = 8
font_size_title = 8

fevs_d1 = [m1_fev_d2_medianUnits,m2_fev_d2_medianUnits]
cis_d1 = [m1_fev_d2_ci,m2_fev_d2_ci]


# fig,axs = plt.subplots(2,1,figsize=(5,10))
# pylustrator.start()

figure,axs = plt.subplots(2,1)
axs = np.ravel(axs)
figure.suptitle('')

col_scheme = ('darkgreen','blue')
# ax.yaxis.grid(True)
xpoints = np.atleast_1d((0,1))
xlabel_fev = ['',''] # ['With\nphotoreceptor\nlayer','Without\nphotoreceptor\nlayer']
axs[0].bar(xpoints,fevs_d1,yerr=cis_d1,align='center',capsize=6,alpha=.7,color=col_scheme,width=0.5,label='with_PR')
axs[0].plot((-1,10),(0,0),color='black',linewidth=0.5)
axs[0].set_xticks([2,3])#(2*np.arange(0,fev_d1_medianUnits_allMdls.shape[0]))
axs[0].set_xticklabels(xlabel_fev)
axs[0].set_ylabel('FEV (%)',fontsize=font_size_ticks)
axs[0].set_title('Trained at photopic light level\nEvaluated at scotopic light level',fontsize=font_size_labels)
# axs[0].set_title('',fontsize=font_size_labels)
axs[0].set_yticks(np.arange(-80,81,20))
axs[0].set_ylim((-100,100))
axs[0].set_xlim((-0.4,1.4))
axs[0].text(0.75,40,'N = %d RGCs' %m1_fev_d2_allUnits.shape[0],fontsize=font_size_ticks)
axs[0].text(0,-90,'With\nphotoreceptor\nlayer',fontsize=font_size_labels,color='darkgreen',horizontalalignment='center')
axs[0].text(1,-90,'Without\nphotoreceptor\nlayer',fontsize=font_size_labels,color='blue',horizontalalignment='center')
axs[0].tick_params(axis='both',labelsize=font_size_labels)

# correlation
fevs_d1 = [m1_predCorr_d2_medianUnits,m2_predCorr_d2_medianUnits]
cis_d1 = [m1_predCorr_d2_ci,m2_predCorr_d2_ci]

# ax.yaxis.grid(True)2
axs[1].bar(xpoints,fevs_d1,yerr=cis_d1,align='center',capsize=6,alpha=.7,color=col_scheme,width=0.5,label='with_PR')
axs[1].set_xticks([2,3])#(2*np.arange(0,fev_d1_medianUnits_allMdls.shape[0]))
axs[1].set_xticklabels(xlabel_fev)

axs[1].set_ylabel('Correlation coefficient',fontsize=font_size_ticks)
axs[1].set_title('',fontsize=font_size_ticks)
axs[1].set_yticks(np.arange(-0.6,1.1,.2))
axs[1].set_ylim((0,1.1))
axs[1].set_xlim((-0.4,1.4))
axs[1].text(.75,.70,'N = %d RGCs' %m1_fev_d2_allUnits.shape[0],fontsize=font_size_ticks)
axs[1].tick_params(axis='both',labelsize=font_size_labels)

# #% start: automatic generated code from pylustrator
# plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
# import matplotlib as mpl
# plt.figure(1).set_size_inches(4.978400/2.54, 7.874000/2.54, forward=True)
# plt.figure(1).axes[0].set_position([0.235566, 0.464252, 0.745593, 0.428679])
# plt.figure(1).axes[0].yaxis.labelpad = -1.197310
# plt.figure(1).axes[0].texts[0].set_position([0.470968, 78.108516])
# plt.figure(1).axes[0].texts[1].set_position([0.118940, -59.778869])
# plt.figure(1).axes[0].texts[2].set_position([0.885489, 10.287264])
# plt.figure(1).axes[0].title.set_position([0.405643, 1.000000])
# plt.figure(1).axes[1].set_position([0.235566, 0.015222, 0.745593, 0.428679])
# plt.figure(1).axes[1].texts[0].set_position([0.470968, 0.960718])
# plt.figure(1).texts[0].set_position([0.596336, 1.017844])
# #% end: automatic generated code from pylustrator
# # plt.show()

# # % start: automatic generated code from pylustrator
# plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
# import matplotlib as mpl
# plt.figure(1).set_size_inches(4.826000/2.54, 9.067800/2.54, forward=True)
# plt.figure(1).axes[0].set_position([0.223009, 0.542521, 0.696954, 0.364957])
# plt.figure(1).axes[0].title.set_position([0.405643, 1.000000])
# plt.figure(1).axes[0].xaxis.labelpad = 0.970704
# plt.figure(1).axes[0].yaxis.labelpad = -1.197310
# plt.figure(1).axes[0].yaxis.labelpad = 0.122937
# plt.figure(1).axes[0].texts[0].set_position([0.470968, 78.108516])
# plt.figure(1).axes[0].title.set_position([0.418281, 1.000000])
# plt.figure(1).axes[1].set_position([0.223009, 0.083532, 0.696954, 0.364957])
# plt.figure(1).axes[1].xaxis.labelpad = -0.199287
# plt.figure(1).axes[1].yaxis.labelpad = -0.097383
# plt.figure(1).texts[0].set_position([0.560244, 1.013822])
# #% end: automatic generated code from pylustrator
# #% start: automatic generated code from pylustrator
# plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
# import matplotlib as mpl
# plt.figure(1).axes[0].yaxis.labelpad = -3.874793
# plt.figure(1).axes[0].texts[0].set_position([0.315025, 79.114039])
# plt.figure(1).axes[0].texts[1].set_position([0.152917, -59.139173])
# plt.figure(1).axes[0].texts[2].set_position([0.827822, 9.546728])
# plt.figure(1).axes[1].texts[0].set_position([0.315025, 0.954397])
# #% end: automatic generated code from pylustrator

# plt.savefig("sample.jpg", dpi=600)
# plt.show()



# %% d. photopic with PR - HIST PLOTS

exp_select = 'retina1'
path_exp = '/home/saad/data/analyses/data_kiersten/'+exp_select+'/8ms_resamp'
mdl_params = 'U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_TR-01'

mdlFolder = os.path.join(path_exp,'photopic-10000_mdl-rieke_s-250_p-40.7_e-879_k-0.01_h-3_b-110_hc-2.64_gd-28_preproc-cones_norm-1_tb-4_RungeKutta_RF-2/CNN_2D/',mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)

val_dataset_2 = 'photopic-10000_mdl-rieke_s-10_p-10_e-20_g-2.5_k-0.01_h-3_b-15.7_hc-5.2_gd-20_preproc-added_norm-1_tb-4_RungeKutta_RF-2'
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0


obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)
obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]

num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
idx_d2_valid = idx_allUnits
idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
idx_d2_valid = idx_allUnits[idx_d2_valid]

m1_fev_d2_allUnits = fev_d2_allUnits*100
m1_fev_d2_medianUnits = np.median(m1_fev_d2_allUnits[idx_d2_valid])
m1_fev_d2_stdUnits = np.std(m1_fev_d2_allUnits[idx_d2_valid])
m1_fev_d2_ci = 1.96*(m1_fev_d2_stdUnits/len(idx_d2_valid)**.5)

m1_predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
m1_predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
m1_predCorr_d2_ci = 1.96*(m1_predCorr_d2_stdUnits/len(idx_d2_valid)**.5)


# % ------------ hist plots  --------------- #
pylustrator.start()

fevs_d1 = [m1_fev_d2_allUnits]
cis_d1 = [m1_fev_d2_ci,0]
bins = np.arange(60,110,2)
# fig,axs = plt.subplots(2,1,figsize=(5,10))
figure,axs = plt.subplots(2,1)
figure.suptitle('')
font_size_ticks = 8
font_size_labels = 8

col_scheme = ('red')
# ax.yaxis.grid(True)
xpoints = np.atleast_1d((0,1))
xlabel_fev = ['With\nphotoreceptor','']
counts, bins = np.histogram(fevs_d1,bins)
axs[0].hist(bins[:-1],bins,weights=counts,color='red')
axs[0].plot((m1_fev_d2_medianUnits,m1_fev_d2_medianUnits),(0,12.5),'--',color='black')
axs[0].set_xticks(np.arange(60,110,10))
axs[0].set_xlabel('FEV (%)',fontsize=font_size_ticks)
axs[0].set_ylabel('Number of RGCs',fontsize=font_size_ticks)
axs[0].set_title('Trained at photopic light level\nEvaluated at photopic light level',fontsize=font_size_ticks)
axs[0].set_xlim((60,100))
# axs[0].text(80,9,'N = %d RGCs' %m1_fev_d2_allUnits.shape[0],fontsize=font_size_ticks)
axs[0].tick_params(axis='both',labelsize=font_size_labels)

# correlation3
fevs_d1 = [predCorr_d2_allUnits]
bins = np.arange(0,1.1,.02)
counts, bins = np.histogram(fevs_d1,bins)
axs[1].hist(bins[:-1],bins,weights=counts,color='red')
axs[1].plot((m1_predCorr_d2_medianUnits,m1_predCorr_d2_medianUnits),(0,24),'--',color='black')
axs[1].plot((0.82,0.82),(0,24),'--',color='orange')
axs[1].set_xticks(np.arange(0,1.1,.1))
axs[1].set_xlabel('Correlation coefficient',fontsize=font_size_ticks)
axs[1].set_ylabel('Number of RGCs',fontsize=font_size_ticks)
axs[1].set_xlim((0.6,1))
axs[1].set_ylim((0,25))
# axs[1].text(0.01,90,'N = %d RGCs' %m1_fev_d2_allUnits.shape[0],fontsize=font_size_ticks)
axs[1].tick_params(axis='both',labelsize=font_size_labels)



# #% start: automatic generated code from pylustrator
# plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
# import matplotlib as mpl
# plt.figure(1).set_size_inches(4.826000/2.54, 9.067800/2.54, forward=True)
# plt.figure(1).axes[0].set_position([0.223009, 0.542521, 0.696954, 0.364957])
# plt.figure(1).axes[0].title.set_position([0.405643, 1.000000])
# plt.figure(1).axes[0].xaxis.labelpad = 0.970704
# plt.figure(1).axes[0].yaxis.labelpad = -1.197310
# plt.figure(1).axes[0].yaxis.labelpad = 0.122937
# plt.figure(1).axes[0].texts[0].set_position([0.470968, 78.108516])
# plt.figure(1).axes[0].title.set_position([0.418281, 1.000000])
# plt.figure(1).axes[1].set_position([0.223009, 0.083532, 0.696954, 0.364957])
# plt.figure(1).axes[1].xaxis.labelpad = -0.199287
# plt.figure(1).axes[1].yaxis.labelpad = -0.097383
# plt.figure(1).texts[0].set_position([0.560244, 1.013822])
# #% end: automatic generated code from pylustrator
# plt.savefig("sample.jpg", dpi=600)
# # plt.show()

