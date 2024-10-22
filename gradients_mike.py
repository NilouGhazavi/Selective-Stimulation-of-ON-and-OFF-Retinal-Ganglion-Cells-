#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:19:59 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import h5py
import numpy as np
import os
import re
from model import utils_si
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import time
from tqdm import tqdm
from jax.tree_util import Partial

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import socket

hostname=socket.gethostname() 
if hostname=='sandwolf':
    base = '/home/saad/data_hdd/'
elif hostname=='sandhound':
    base = '/home/saad/postdoc_db/'

base = '/home/saad/postdoc_db/'
# base = '/home/saad/data_hdd/'


from scipy.stats import wilcoxon
import scipy
import gc

import csv
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])

from model.load_savedModel import load
import model.data_handler
from model.data_handler import load_data, prepare_data_cnn2d, prepare_data_cnn3d, prepare_data_convLSTM, prepare_data_pr_cnn2d,merge_datasets,isintuple
from model.data_handler_mike import load_h5Dataset
from model.performance import getModelParams, model_evaluate,model_evaluate_new,paramsToName, get_weightsDict, get_weightsOfLayer,estimate_noise
from model import metrics
from model import featureMaps
from model.models import modelFileName
from model.train_model import chunker
import model.gradient_tools
from model.featureMaps import spatRF2DFit, get_strf, decompose
from model.train_model import chunker


# from pyret.filtertools import sta, decompose
# import seaborn

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Input, Reshape
from model.train_model import chunker

import glob

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.regularizers import l2

import jax
from jax import numpy as jnp



def load_model_from_path(path_model):
    mdlFolder = os.path.split(path_model)[-1]
    mdl_epochList = np.sort(glob.glob(os.path.join(path_model,'*.index')))
    lastEpoch = mdl_epochList[-1].split('.index')[0]
    fname_bestWeight = os.path.split(lastEpoch)[-1]
    mdl = load(os.path.join(path_model,mdlFolder))
    mdl.load_weights(os.path.join(path_model,fname_bestWeight))
    
    return mdl


def jax_add(a,b):
    @jax.jit
    def oper_add(a1,b1):
        return a1+b1
    
    out = []
    for i in range(len(a)):
        rgb = oper_add(a[i],b[i])
        out.append(rgb)
    
    out = np.array(out)

    return out

def jax_subtract(a,b):
    @jax.jit
    def oper_subtract(a1,b1):
        return jnp.subtract(a1,b1)
    
    out = []
    for i in range(len(a)):
        rgb = oper_subtract(a[i],b[i])
        out.append(rgb)
    out = np.array(out)
    return out

# %% Load the model and get performance

path_main = '/home/saad/data_hdd/analyses/data_mike/20230725C/models'
# [CNN,PR-CNN]
dict_lrs = {
    'NATSTIM0': ['0.0001', '1e-05'],
    'NATSTIM1': ['0.0001', '0.0001'],
    'NATSTIM2': ['0.0001', '0.0001'],
    'NATSTIM3': ['0.001', '1e-05'],
    'NATSTIM4': ['0.0001', '1e-05'],
    'NATSTIM5': ['0.0001', '0.0001'],
    'NATSTIM6': ['0.0001', '0.0001'],
    'NATSTIM7': ['0.0001', '1e-05'],
    'NATSTIM8': ['0.0001', '1e-05']
    }

stims = list(dict_lrs.keys())

i=2
stim_select = stims[i]
path_cnn = os.path.join(path_main,stim_select+'_CORR_mesopic-Rstar_f4_8ms/finetuning_crossval/CNN2D/U-57_T-080_C1-10-11_C2-15-07_C3-20-07_BN-1_MP-2_LR-'+dict_lrs[stim_select][0]+'_TRSAMPS--01_TR-01')

fname_data_train_val_test = '/home/saad/postdoc_db/analyses/data_mike/20230725C/datasets/20230725C_dataset_train_val_test_'+stim_select+'_CORR_mesopic-Rstar_f4_8ms.h5'


rgb = load_h5Dataset(fname_data_train_val_test,nsamps_val=0.5,nsamps_test=0.1,nsamps_train=0.05,LOAD_TR=True)
data_train = rgb[0]
data_val = rgb[1];
parameters = rgb[5]
data_quality = rgb[3]
t_frame = parameters['t_frame']     # time in ms of one frame/sample 
uname_all = data_quality['uname_selectedUnits']


rate_sameStim_trials = data_val.y_trials
rate_sameStim_trials = np.squeeze(np.moveaxis(rate_sameStim_trials,-1,0))
_, fracExVar_allUnits, _, corr_allUnits = model_evaluate(rate_sameStim_trials,None,0,RR_ONLY=True)
retinalReliability_fev_allrgcs = fracExVar_allUnits
retinalReliability_fev = np.round(np.nanmedian(fracExVar_allUnits),2)

data_train =  prepare_data_cnn2d(data_train,80,np.arange(data_train.y.shape[1]))
data_val_cnn = prepare_data_cnn2d(data_val,80,np.arange(data_val.y.shape[1]))

obs_noise = data_quality['var_noise'] #estimate_noise(data_test.y_trials)


idx_unitsToTake = np.arange(0,57)


# tf.compat.v1.disable_eager_execution()
mdl_cnn = load_model_from_path(path_cnn)
pred_rate_cnn = mdl_cnn.predict(data_val_cnn.X)


# CNN
fev_cnn_allUnits, fracExplainableVar, predCorr_cnn_allUnits, rrCorr_cnn_allUnits = model_evaluate_new(
    data_val_cnn.y.copy(),pred_rate_cnn.copy(),0,RR_ONLY=False,lag = 0,obs_noise=obs_noise.copy())

idx_allUnits = np.arange(fev_cnn_allUnits.shape[0])
idx_d1_valid = idx_allUnits
# idx_d1_valid = fev_cnn_allUnits<1.1
# idx_d1_valid = idx_allUnits[idx_d1_valid]
idx_on = np.intersect1d(np.arange(0,27),idx_d1_valid)
idx_off = np.intersect1d(np.arange(27,57),idx_d1_valid)


print('N = %d RGCs'%len(idx_d1_valid))



fev_cnn_medianUnits = np.nanmedian(fev_cnn_allUnits[idx_d1_valid])
print('FEV_CNN = %0.2f' %(fev_cnn_medianUnits*100))
fev_cnn_stdUnits = np.nanstd(fev_cnn_allUnits[idx_d1_valid])
fev_cnn_ci = 1.96*(fev_cnn_stdUnits/len(idx_d1_valid)**.5)

predCorr_cnn_medianUnits = np.nanmedian(predCorr_cnn_allUnits[idx_d1_valid])
print('R = %0.2f' %predCorr_cnn_medianUnits)
predCorr_cnn_stdUnits = np.nanstd(predCorr_cnn_allUnits[idx_d1_valid])
predCorr_cnn_ci = 1.96*(predCorr_cnn_stdUnits/len(idx_d1_valid)**.5)

# %% Select RGCs and Training data
"""
fname_rfs = '/home/saad/postdoc_db/analyses/data_mike/20230725C/datasets/20230725C_RFs_CB_allLightLevels.h5'
with h5py.File(fname_rfs,'r') as f:
    rf_coords_all = np.array(f['mesopic']['rf_coords_all'])

for i in range(40,rf_coords_all.shape[0]):
    plt.plot(rf_coords_all[i,:,0],rf_coords_all[i,:,1])
    plt.text(np.mean(rf_coords_all[i,:,0]),np.mean(rf_coords_all[i,:,1]),str(i))
plt.xlim([0,100]);plt.ylim([0,80]);plt.gca().invert_yaxis()


fev_cnn_allUnits[[2,4,5,23,33,34,53,56]]
"""

# idx_unitsToExtract = idx_allUnits
# idx_unitsToExtract = np.argsort(-1*fev_cnn_allUnits)
idx_unitsToExtract = [2,4,5,23,33,34,53,56] #[2,4,5,10,12,27,31,30,49,50]
fev_unitsToExtract=fev_cnn_allUnits[idx_unitsToExtract]
uname_unitsToExtract = uname_all[idx_unitsToExtract]
print(uname_unitsToExtract)


rgb = load_h5Dataset(fname_data_train_val_test,nsamps_val=0.5,nsamps_test=0.1,nsamps_train=-1,LOAD_TR=True)
data_train = rgb[0]
# Only select 1 trial
X = data_train.X[:,:,:,:,5:6]
y = np.mean(data_train.y,axis=-1)
data_train = Exptdata(X,y)
data_train =  prepare_data_cnn2d(data_train,80,np.arange(data_train.y.shape[1]))


# %% Compute gradients for all the datasets (and models?)
"""
Because extracting gradients require gpu memory, we have to extract gradients
in batches. Each batch is of batch_size. For efficient processing, we first
calculate gradients for each batch, then those gradients are stored in a list.
The list iterates over batches. Then when we have iterated over all the batches
i.e. we have a list the same size as total_batches, we concatenate everything into
a single large matrix.

This section outputs data_alldsets. Structure is:
    data_alldsets
        ----- dataset_name
            ------ grads_all --> [n_outputUnits,temporal_width,pixels_y,pixels_x,samples]
            ------ stim_mat -->  [x_samples,temporal_width,pixels_y,pixels_x]
            
Gradients are computed within GradientTape framework. This allows TF to 'record'
relevant operations in the forward pass. Then during backward pass, TF traverses
this list of operations in reverse order to compute gradients.
"""

path_grads = '/home/saad/data/analyses/gradients_mike/'
temporal_width_grads = 80
select_mdl = 'CNN_2D' #'PRFR_CNN2D_RODS' #'CNN_2D_NORM'
save_grads = True

mdl_totake = mdl_cnn

tempWidth_inp = mdl_totake.input.shape[1]
weights_dense_orig = mdl_totake.layers[-2].get_weights()

counter_gc = 0
n_units = len(idx_unitsToExtract)
# idx_unitsToExtract = np.arange(n_units)

data = data_train #data_alldsets[d]['raw']
nsamps = len(data.X)

dataset_eval = [stim_select+'_CORR_mesopic-Rstar_f4_8ms',]
d = dataset_eval[0]
for d in dataset_eval:
    if save_grads==True:
        fname_gradsFile = os.path.join(path_grads,'grads_'+select_mdl+'_'+d+'_'+str(nsamps)+'_u-'+str(n_units)+'.h5')
        if os.path.exists(fname_gradsFile):
            fname_gradsFile = fname_gradsFile[:-3]+'_1.h5'

    batch_size = 256
    total_batches = int(np.floor((nsamps/batch_size)))
    
    i = 50
    grads_shape = (n_units,None,temporal_width_grads,data.X[0].shape[1],data.X[0].shape[2])
    stim_shape = (None,)
    
    t_start = time.time()
    if save_grads==True:
        f_grads = h5py.File(fname_gradsFile,'a')
        grp = model.gradient_tools.init_GradDataset(f_grads,select_mdl,d,grads_shape,stim_shape,batchsize=batch_size)
    
    for i in tqdm(range(0,total_batches)):
        counter_gc+=1
        print (' List: Batch %d of %d'%(i+1,total_batches))
        idx_chunk = np.arange(i*batch_size,(i+1)*batch_size)
        data_select_X = np.array(data.X[idx_chunk[0]:idx_chunk[-1]+1])[:,-tempWidth_inp:]
        stim_chunk = None #np.array(data_select_X).astype('float16')
        
        inp = tf.Variable(data_select_X, dtype=tf.float32, name='input')

        grads_chunk_allUnits = np.zeros((len(idx_unitsToExtract),batch_size,temporal_width_grads,data.X[0].shape[-2],data.X[0].shape[-1]),dtype='float32')
        t_batch_start = time.time()
        u=0
        for u in range(len(idx_unitsToExtract)):
            idx_unitToModel = np.atleast_1d(idx_unitsToExtract[u])
            
            n_out = idx_unitToModel.shape[0]
            y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3))(mdl_totake.layers[-3].output)
            outputs = Activation('softplus',dtype='float32',name='new_activation')(y)
            
            mdl_new = Model(mdl_totake.inputs,outputs)
            
            a = weights_dense_orig[0][:,idx_unitToModel]
            b = weights_dense_orig[1][idx_unitToModel]
            weights_dense_new = [a,b]
            
            mdl_new.layers[-2].set_weights(weights_dense_new)
            
            
            with tf.GradientTape(persistent=False,watch_accessed_variables=True) as tape:
                out = mdl_new(inp,training=False)
            grads_chunk = tape.gradient(out, inp)
            
            grads_chunk = grads_chunk[:,-temporal_width_grads:,:,:]
            grads_chunk = np.array(grads_chunk)
            
            grads_chunk_allUnits[u] = grads_chunk
    
        if save_grads==True:
            model.gradient_tools.append_GradDataset(f_grads,grp,grads_chunk_allUnits,stim_chunk)
        
        if counter_gc == 250:
            _ = gc.collect()
            counter_gc = 0

        t_batch = time.time()-t_batch_start
        print(t_batch/60)

    t_dur = time.time()-t_start
    print(t_dur/60)
    if save_grads==True:
        # grp.create_dataset('idx_data',data=data_alldsets[d]['idx_samps'])
        grp.create_dataset('unames',data=uname_unitsToExtract.astype('bytes'))
        grp.create_dataset('fev',data=fev_unitsToExtract)
    f_grads.close()


# %% Arrange gradients

path_gradFiles = '/home/saad/data/analyses/gradients_mike/'

select_mdl = 'CNN_2D' #'PRFR_CNN2D_RODS' #'CNN_2D_NORM'
dataset_eval = [stim_select+'_CORR_mesopic-Rstar_f4_8ms',]
d = dataset_eval[0]
data = data_train
nsamps = len(data.X)
n_units = len(idx_unitsToExtract)

fname_gradsFile = os.path.join(path_gradFiles,'grads_'+select_mdl+'_'+d+'_'+str(nsamps)+'_u-'+str(n_units)+'.h5')
f_grads = h5py.File(fname_gradsFile,'r')

data_select = np.arange(2690,3220)

grads_chunk_allUnits = np.array(f_grads[select_mdl]['NATSTIM2_CORR_mesopic-Rstar_f4_8ms']['grads'][:,data_select[0]:data_select[-1]])
stim = np.asarray(data.X[data_select[0]:data_select[-1]])
y_orig = np.asarray(data.y[data_select[0]:data_select[-1]])
# stim_vec = stim.reshape(stim.shape[0],-1)

rgc_shut = [33,34,53,56]
rgc_fixed = [2,4,5,23]

# rgc_shut = [27,31,30,49,50]
# rgc_fixed = [2,4,5,10,12] #[10]

_,idx_rgc_shut,_ = np.intersect1d(idx_unitsToExtract,rgc_shut,return_indices=True)
grads_rgc_shut = grads_chunk_allUnits[idx_rgc_shut]
grads_rgc_shut_vec = grads_rgc_shut.reshape(grads_rgc_shut.shape[0],grads_rgc_shut.shape[1],-1)
grads_rgc_shut_mean = np.mean(grads_rgc_shut,axis=0)

_,idx_rgc_fixed,_ = np.intersect1d(idx_unitsToExtract,rgc_fixed,return_indices=True)
grads_rgc_fixed = grads_chunk_allUnits[idx_rgc_fixed]
grads_rgc_fixed_mean = np.array(jnp.mean(grads_rgc_fixed,axis=0))
grads_rgc_fixed_vec = grads_rgc_fixed.reshape(grads_rgc_fixed.shape[0],grads_rgc_fixed.shape[1],-1)

# RFs
fname_rfs = '/home/saad/postdoc_db/analyses/data_mike/20230725C/datasets/20230725C_RFs_CB_allLightLevels.h5'
with h5py.File(fname_rfs,'r') as f:
    rf_coords_all = np.array(f['mesopic']['rf_coords_all'])
rfs_shut = rf_coords_all[rgc_shut]
rfs_fixed = rf_coords_all[rgc_fixed]


# %% Orthonormal basis for Unaffected RGCs

@jax.jit
def gram_schmidt_basis(vectors):
    basis = []
    for v in vectors:
        # Orthogonalize v against the existing basis vectors
        for b in basis:
            v = v - jnp.dot(v, b) / jnp.dot(b, b) * b
        # Normalize v and add it to the basis
        basis.append(v / jnp.linalg.norm(v))
    
    return basis

@jax.jit
def projection_onto_orthogonal_basis(A, basis):
    # Compute the projection of A onto each basis vector
    projections = [jnp.dot(A, b) / jnp.dot(b, b) * b for b in basis]
    # Sum up the projections to get the projection onto the orthogonal subspace
    projections = jnp.array(projections)
    orthogonal_projection = jnp.sum(projections, axis=0)
    
    return orthogonal_projection

grad_shutProjFixed_allUnits_vec = []
i=0;j=0;
for i in tqdm(range(grads_rgc_shut_vec.shape[0])):        # iterate over units
    grad_proj_vec = []
    for j in range(grads_rgc_shut_vec.shape[1]):    # iterate over stim
        A = grads_rgc_shut_vec[i,j]
        B = grads_rgc_fixed_vec[:,j]
        orthogonal_basis = gram_schmidt_basis(B)
        projection_orthogonal = projection_onto_orthogonal_basis(A, orthogonal_basis)
        grad_proj_vec.append(projection_orthogonal)

    grad_shutProjFixed_allUnits_vec.append(grad_proj_vec)
    
grad_shutProjFixed_allUnits_vec = jnp.array(grad_shutProjFixed_allUnits_vec)


grad_toadjust = jax_subtract(grads_rgc_shut_vec,grad_shutProjFixed_allUnits_vec)
grad_toadjust_mean = jnp.nanmean(grad_toadjust,axis=0)
grad_toadjust_mean = grad_toadjust_mean.reshape(grad_toadjust_mean.shape[0],*stim.shape[1:])

del grad_shutProjFixed_allUnits_vec, grad_toadjust
# %% Apply adjustment to stimulus and get new predictions
step = 30
rgb = jnp.concatenate((stim[None,:,:,:,:],-1*grad_toadjust_mean[None,:,:,:,:]*step),axis=0)
stim_modified = jnp.sum(rgb,axis=0)
stim_modified = np.array(stim_modified)


# Orig stim response
tf.compat.v1.disable_eager_execution()
mdl_cnn = load_model_from_path(path_cnn)
mdl_totake = mdl_cnn
y_pred = mdl_totake.predict(stim)
y_pred_mod = mdl_totake.predict(stim_modified)

fig,axs=plt.subplots(2,4,figsize=(25,7));#axs=np.ravel(axs)
fig.suptitle('step = '+str(step))
for i in range(len(rgc_shut)):
    axs[0,i].plot(y_orig[:,rgc_shut[i]],'lightgray');axs[0,i].plot(y_pred[:,rgc_shut[i]]);axs[0,i].plot(y_pred_mod[:,rgc_shut[i]],'--');axs[0,i].set_title(uname_all[rgc_shut[i]])
    axs[1,i].plot(y_orig[:,rgc_fixed[i]],'lightgray');axs[1,i].plot(y_pred[:,rgc_fixed[i]]);axs[1,i].plot(y_pred_mod[:,rgc_fixed[i]],'--');axs[1,i].set_title(uname_all[rgc_fixed[i]])

# %% Plot stimulus
idx_temp_peak = 67 #np.argmax(np.abs(tempRF_avg))
idx_peakFromSpkOnset = -13#stim.shape[1]-idx_temp_peak

LINEWIDTH = .5
img_idx=400;

fig,axs=plt.subplots(2,3,figsize=(20,10));axs=np.ravel(axs)
vmin_img,vmax_img = stim_modified[img_idx,idx_temp_peak].min(),stim_modified[img_idx,idx_temp_peak].max()
h0 = axs[0].imshow(stim[img_idx,idx_temp_peak],cmap='gray',vmin=vmin_img,vmax=vmax_img);axs[0].set_title('Stim');plt.colorbar(h0,fraction=0.035,pad=0.02);axs[0].set_ylim([74,0]);
axs[0].plot(rfs_shut[:,:,0].T,rfs_shut[:,:,1].T,'r',linewidth=LINEWIDTH);axs[1].set_ylim([74,0]);axs[0].plot(rfs_fixed[:,:,0].T,rfs_fixed[:,:,1].T,'b',linewidth=LINEWIDTH);axs[4].set_ylim([74,0]);
h1=axs[1].imshow(grads_rgc_shut_mean[img_idx,idx_temp_peak],cmap='gray');axs[1].set_title('LSTA unit: '+uname_unitsToExtract[idx_rgc_shut]);plt.colorbar(h1,fraction=0.035,pad=0.02)
axs[1].plot(rfs_shut[:,:,0].T,rfs_shut[:,:,1].T,'r');axs[1].set_ylim([74,0]);
h2=axs[2].imshow(grads_rgc_fixed_mean[img_idx,idx_temp_peak],cmap='gray');axs[2].set_title('LSTA unit: '+uname_unitsToExtract[idx_rgc_fixed]);plt.colorbar(h2,fraction=0.035,pad=0.02)
axs[2].plot(rfs_fixed[:,:,0].T,rfs_fixed[:,:,1].T,'b');axs[2].set_ylim([74,0]);
h3=axs[3].imshow(stim_modified[img_idx,idx_temp_peak],cmap='gray',vmin=vmin_img,vmax=vmax_img);axs[3].set_title('New stim');plt.colorbar(h3,fraction=0.035,pad=0.02)
h4=axs[4].imshow(step*grad_toadjust_mean[img_idx,idx_temp_peak],cmap='gray');axs[4].set_title('Gradient adjusted');plt.colorbar(h4,fraction=0.035,pad=0.02)
axs[4].plot(rfs_shut[:,:,0].T,rfs_shut[:,:,1].T,'r',linewidth=LINEWIDTH);axs[1].set_ylim([74,0]);axs[4].plot(rfs_fixed[:,:,0].T,rfs_fixed[:,:,1].T,'b',linewidth=LINEWIDTH);axs[4].set_ylim([74,0]);
axs[5].axis('off')

# %% Plot High res stim for saving
import cv2 as cv
rsz_fac = 8

img = cv.imread('/home/saad/postdoc_db/proposals/gradients/raw/img.png')[:,:,0]/255
img = (img * (stim[img_idx,idx_temp_peak].max() - stim[img_idx,idx_temp_peak].min())) + stim[img_idx,idx_temp_peak].min()

grad_toadjust_mean_re = np.array(grad_toadjust_mean[img_idx,idx_temp_peak]).astype('float32')
grad_toadjust_mean_re = step*cv.resize(grad_toadjust_mean_re,[800,600])
img_modified = img - grad_toadjust_mean_re
rfs_shut_re = rfs_shut*rsz_fac
rfs_fixed_re = rfs_fixed*rsz_fac


fig,axs=plt.subplots(1,3,figsize=(20,10));axs=np.ravel(axs)
vmin_img,vmax_img = stim_modified[img_idx,idx_temp_peak].min(),stim_modified[img_idx,idx_temp_peak].max()
h0 = axs[0].imshow(img,cmap='gray',vmin=vmin_img,vmax=vmax_img);axs[0].set_title('Stim');plt.colorbar(h0,fraction=0.035,pad=0.02)
axs[0].plot(rfs_shut_re[:,:,0].T,rfs_shut_re[:,:,1].T,'r',linewidth=LINEWIDTH);axs[0].plot(rfs_fixed_re[:,:,0].T,rfs_fixed_re[:,:,1].T,'b',linewidth=LINEWIDTH)
axs[0].set_ylim([600,0]);

h2=axs[1].imshow(-grad_toadjust_mean_re,cmap='gray',vmin=vmin_img-.22,vmax=vmax_img);axs[1].set_title('Gradient adjusted');plt.colorbar(h2,fraction=0.035,pad=0.02)
axs[1].plot(rfs_shut_re[:,:,0].T,rfs_shut_re[:,:,1].T,'r',linewidth=LINEWIDTH);axs[1].plot(rfs_fixed_re[:,:,0].T,rfs_fixed_re[:,:,1].T,'b',linewidth=LINEWIDTH)
axs[1].set_ylim([600,0])

h1=axs[2].imshow(img_modified,cmap='gray',vmin=vmin_img,vmax=vmax_img);axs[2].set_title('New stim');plt.colorbar(h1,fraction=0.035,pad=0.02)
axs[2].plot(rfs_shut_re[:,:,0].T,rfs_shut_re[:,:,1].T,'r',linewidth=LINEWIDTH);axs[2].plot(rfs_fixed_re[:,:,0].T,rfs_fixed_re[:,:,1].T,'b',linewidth=LINEWIDTH)
axs[2].set_ylim([600,0])


fname_fig = '/home/saad/postdoc_db/proposals/gradients/raw/stims_resized'
fig.savefig(fname_fig+'.png',dpi=200)
fig.savefig(fname_fig+'.svg',dpi=600)


# %% Plot Movie
idx_temp_peak_sel = -13

vmin_img,vmax_img = np.nanmin(stim_modified[:,idx_temp_peak_sel]),np.nanmax(stim_modified[:,idx_temp_peak_sel])
for i in range(stim_modified.shape[0]):
    fig,axs=plt.subplots(1,3,figsize=(20,10));axs=np.ravel(axs)
    # vmin_img,vmax_img = img_modified[:,idx_temp_peak].min(),img_modified[i,idx_temp_peak].max()
    h0 = axs[0].imshow(stim[i,idx_temp_peak_sel],cmap='gray',vmin=vmin_img,vmax=vmax_img);axs[0].set_title('Stim original');axs[0].set_ylim([74,0]);
    axs[0].plot(rfs_shut[:,:,0].T,rfs_shut[:,:,1].T,'r',linewidth=LINEWIDTH);
    axs[0].plot(rfs_fixed[:,:,0].T,rfs_fixed[:,:,1].T,'b',linewidth=LINEWIDTH)
    axs[0].set_ylim([74,0]);
    
    
    h1 = axs[1].imshow(stim_modified[i,idx_temp_peak_sel],cmap='gray',vmin=vmin_img,vmax=vmax_img);axs[1].set_title('Stim modified');axs[1].set_ylim([74,0]);
    axs[1].plot(rfs_shut[:,:,0].T,rfs_shut[:,:,1].T,'r',linewidth=LINEWIDTH);
    axs[1].plot(rfs_fixed[:,:,0].T,rfs_fixed[:,:,1].T,'b',linewidth=LINEWIDTH)
    axs[1].set_ylim([74,0]);

    h2 = axs[2].imshow(step*grad_toadjust_mean[i,idx_temp_peak_sel],cmap='gray',vmin=vmin_img,vmax=vmax_img);axs[2].set_title('Stim modified');axs[2].set_ylim([74,0]);
    axs[2].plot(rfs_shut[:,:,0].T,rfs_shut[:,:,1].T,'r',linewidth=LINEWIDTH);
    axs[2].plot(rfs_fixed[:,:,0].T,rfs_fixed[:,:,1].T,'b',linewidth=LINEWIDTH)
    axs[2].set_ylim([74,0]);

    
    plt.show()


# %% Save Movie Hi-Res
from model.data_handler import rolling_window
import ffmpeg
from skimage.transform import resize

fname_hires = '/home/saad/postdoc_db/analyses/data_mike/20230725C/datasets/frames_hires.h5'

f = h5py.File(fname_hires,'r')
stim_hr = np.array(f['frames'])
f.close()
stim_hr = rolling_window(stim_hr,80)

idx_temp_peak_sel = -13

stim_hr = stim_hr[114:-1,idx_temp_peak_sel]
grad_toadjust_mean_hr = np.array(grad_toadjust_mean[:,idx_temp_peak_sel]).astype('float32')
grad_toadjust_mean_hr[grad_toadjust_mean_hr==np.nan] = 0
grad_toadjust_mean_hr[149:156]=0
# grad_toadjust_mean_hr = np.moveaxis(grad_toadjust_mean_hr,0,-1)
grad_toadjust_mean_hr = step*resize(grad_toadjust_mean_hr,[grad_toadjust_mean_hr.shape[0],600,800])
stim_hr_modified = stim_hr - grad_toadjust_mean_hr

# %

def create_movie(movie_mat,fname_video,vmin_mov,vmax_mov):
    movie_mat = ((movie_mat - vmin_mov)/(vmax_mov-vmin_mov)*255)
    movie_mat = movie_mat.astype('uint8')


    _,h,w = movie_mat.shape
    framerate = 30
    ffmpeg_process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='gray8', s='{}x{}'.format(w, h))
            .output(fname_video, pix_fmt='yuv420p', r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    

    frame  = movie_mat[0]
    for frame in movie_mat:
        ffmpeg_process.stdin.write(frame.tobytes())
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    

vmin_mov = np.nanmin(stim_hr_modified)
vmax_mov = np.nanmax(stim_hr_modified)

movie_mat = stim_hr
fname_video = '/home/saad/postdoc_db/proposals/gradients/raw/grads_01_StimOrig_vid.mp4'
create_movie(movie_mat,fname_video,vmin_mov,vmax_mov)

movie_mat = stim_hr_modified
fname_video = '/home/saad/postdoc_db/proposals/gradients/raw/grads_03_StimModified_vid.mp4'
create_movie(movie_mat,fname_video,vmin_mov,vmax_mov)

movie_mat = grad_toadjust_mean_hr
fname_video = '/home/saad/postdoc_db/proposals/gradients/raw/grads_02_StimChange_vid.mp4'
create_movie(movie_mat,fname_video,grad_toadjust_mean_hr.min(),grad_toadjust_mean_hr.max())

buff = 1e-6*np.ones((stim_hr.shape[0],600,10))
movie_mat =np.concatenate((stim_hr,buff,stim_hr_modified),axis=2)
fname_video = '/home/saad/postdoc_db/proposals/gradients/raw/grads_04_StimOrig_Modified_vid.mp4'
create_movie(movie_mat,fname_video,vmin_mov,vmax_mov)


ctr=-1
for frame in movie_mat:
    ctr=ctr+1
    plt.imshow(frame,cmap='gray');plt.title(str(ctr));plt.show()
    
# %%Get LSTAs
spatRF_allImg = np.zeros((grads_rgc_shut.shape[1],grads_rgc_shut.shape[-2],grads_rgc_shut.shape[-1]));spatRF_allImg[:]=np.nan
tempRF_allImg = np.zeros((grads_rgc_shut.shape[1],grads_rgc_shut.shape[2]));tempRF_allImg[:]=np.nan


@jax.jit
def get_rf(grad_img):
    spatRF, tempRF = model.featureMaps.decompose(grads_rgc_shut[u,select_img,:,:,:])
    mean_rfCent = np.abs(np.nanmean(spatRF))
    spatRF = spatRF/mean_rfCent
    tempRF = tempRF*mean_rfCent
    tempRF = tempRF/tempRF.max()    
    return spatRF,tempRF


u=0
for i in range(spatRF_allImg.shape[0]):
    select_img = i #768 #712
    spatRF, tempRF = get_rf(grads_rgc_fixed[u,select_img,:,:,:])
    # spatRF, tempRF = model.featureMaps.decompose(grads_rgc_shut[u,select_img,:,:,:])

    spatRF_allImg[i] = spatRF
    tempRF_allImg[i] = tempRF

spatRF_avg = np.nanmean(spatRF_allImg,axis=0)
tempRF_avg = np.nanmean(tempRF_allImg,axis=0)

plt.imshow(spatRF_avg,cmap='gray');plt.show()
plt.plot(tempRF_avg);plt.show()

idx_temp_peak = 67 #np.argmax(np.abs(tempRF_avg))
idx_peakFromSpkOnset = -13#stim.shape[1]-idx_temp_peak


plt.imshow(grads_rgc_shut[u,15,-13,:,:],cmap='gray');plt.show()


# %% JAX gradients
"""
trainable_variables = [var.numpy() for var in mdl_new.trainable_variables]
weights_dict = get_weightsDict(mdl_new)

def model_output(mdl_new,data_select_X):
    return mdl_new.predict(data_select_X)

def test_fn(x,y):
    return  x ** 2 + x * y + y ** 2

x = -2.0
y = -3.0
grad_fn = jax.grad(model_output,argnums=1,has_aux=False)
grads = grad_fn(mdl_new,data_select_X)



def wrapped_model(inp):
    return jnp.array(mdl_new(inp,training=False))

pred_rate = jax.jit(wrapped_model)
pred_rate = pred_rate(data_select_X)

grad_fn = jax.jit(jax.grad(wrapped_model,argnums=0))
grads = grad_fn(data_select_X)
"""

