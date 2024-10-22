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
import socket
hostname=socket.gethostname()
if hostname=='sandwolf':
    base = '/home/saad/data_hdd/'
elif hostname=='sandhound':
    base = '/home/saad/postdoc_db/'

base = '/home/saad/postdoc_db/'
# base = '/home/saad/data_hdd/'



data_pers = 'mike' 
expDate = '20230725C'
expFold = expDate
subFold = '' 
dataset = ('NATSTIM3_CORR_mesopic-Rstar_f4_8ms',)#'NATSTIM3_CORR_mesopic-Rstar_f4_8ms  CB_CORR_mesopic-Rstar_f4_8ms

idx_unitsToTake = 0#idx_units_ON_train #[0] #idx_units_train
select_rgctype=0
mdl_subFold = 'finetuning'
mdl_name = 'PRFR_CNN2D' #PRFR_CNN2D'  #CNN_2D_NORM2' #'
pr_params_name = 'fr_rods_fixed'#'prln_cones_trainable' #'mike_phot_beta0'
path_existing_mdl = '/home/saad/postdoc_db/analyses/data_mike/20230725C/models/CB_CORR_mesopic-Rstar_f4_8ms/PRFR_CNN2D/fr_rods_fixed/U-57_P-080_T-078_C1-10-15_C2-15-11_C3-25-11_BN-1_MP-2_LR-0.001_TRSAMPS--01_TR-01'#
transfer_mode = 'finetuning'
info = ''
idxStart_fixedLayers = 0#1
idxEnd_fixedLayers = -1#15   #29 dense; 28 BN+dense; 21 conv+dense; 15 second conv; 8 first conv
CONTINUE_TRAINING = 1

lr = 1e-05
lr_fac = 1# how much to divide the learning rate when training is resumed
use_lrscheduler=1
lrscheduler='constant' #dict(scheduler='stepLR',drop=0.01,steps_drop=20,initial_lr=lr)
USE_CHUNKER=1
pr_temporal_width = 80
temporal_width=78
thresh_rr=0
chans_bp = 0
chan1_n=10
filt1_size=15
filt1_3rdDim=0
chan2_n=15
filt2_size=11
filt2_3rdDim=0
chan3_n=25
filt3_size=11
filt3_3rdDim=0
chan4_n=0
filt4_size=0
filt4_3rdDim=0
nb_epochs=110#42         # setting this to 0 only runs evaluation
bz_ms=1000#10000#5000
BatchNorm=1
MaxPool=2
runOnCluster=0
num_trials=1

BatchNorm_train = 1
saveToCSV=1
trainingSamps_dur = -1#20 #-1 #0.05 # minutes per dataset
validationSamps_dur=0.03
testSamps_dur=0.03
USE_WANDB = 1


dataset_nameForPaths = ''
for i in range(len(dataset)):
    dataset_nameForPaths = dataset_nameForPaths+dataset[i]+'+'

dataset_nameForPaths = dataset_nameForPaths[:-1]

path_model_save_base = os.path.join(base,'analyses/data_'+data_pers+'/',expDate,subFold,'models',dataset_nameForPaths,mdl_subFold)
path_dataset_base = os.path.join('/home/saad/postdoc_db/analyses/data_'+data_pers+'/',expDate,subFold)

fname_data_train_val_test = ''
i=0
for i in range(len(dataset)):
    name_datasetFile = expDate+'_dataset_train_val_test_'+dataset[i]+'.h5'
    fname_data_train_val_test = fname_data_train_val_test+os.path.join(path_dataset_base,'datasets',name_datasetFile) + '+'
fname_data_train_val_test = fname_data_train_val_test[:-1]

c_trial = 1

if path_existing_mdl=='' and idxStart_fixedLayers>0:
    raise ValueError('Transfer learning set. Define existing model path')

# %
for c_trial in range(1,num_trials+1):
    model_performance,mdl = run_model(expDate,mdl_name,path_model_save_base,fname_data_train_val_test,path_dataset_base=path_dataset_base,saveToCSV=saveToCSV,runOnCluster=0,
                            temporal_width=temporal_width, pr_temporal_width=pr_temporal_width, pr_params_name=pr_params_name,
                            thresh_rr=thresh_rr,chans_bp=chans_bp,
                            chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim,
                            chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim,
                            chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim,
                            chan4_n=chan4_n, filt4_size=filt4_size, filt4_3rdDim=filt4_3rdDim,
                            nb_epochs=nb_epochs,bz_ms=bz_ms,
                            BatchNorm=BatchNorm,BatchNorm_train = BatchNorm_train,MaxPool=MaxPool,c_trial=c_trial,USE_CHUNKER=USE_CHUNKER,
                            path_existing_mdl = path_existing_mdl, idxStart_fixedLayers=idxStart_fixedLayers, idxEnd_fixedLayers=idxEnd_fixedLayers,
                            CONTINUE_TRAINING=CONTINUE_TRAINING,info=info,
                            trainingSamps_dur=trainingSamps_dur,validationSamps_dur=validationSamps_dur,idx_unitsToTake=idx_unitsToTake,
                            lr=lr,lr_fac=lr_fac,use_lrscheduler=use_lrscheduler,lrscheduler=lrscheduler,USE_WANDB=USE_WANDB)
    
plt.plot(model_performance['fev_medianUnits_allEpochs']);plt.ylabel('FEV');plt.xlabel('Epochs')
print('FEV = %0.2f' %(np.nanmax(model_performance['fev_medianUnits_allEpochs'])*100))



# %% Evaluate several models in loop
import numpy as np
import matplotlib.pyplot as plt
import os
from model.performance import getModelParams
from run_models import run_model
import socket
hostname=socket.gethostname()
if hostname=='sandwolf':
    base = '/home/saad/data_hdd/'
elif hostname=='sandhound':
    base = '/home/saad/postdoc_db/'

min_nepochs = 70

data_pers = 'mike'
expDate = '20230725C'
expFold = expDate
subFold = ''
dataset = ('CB_photopic-Rstar','NATSTIM_photopic-Rstar')
path_existing_mdl = ''
info = ''


mdl_subFold = 'cluster'
mdl_name = 'CNN_2D_NORM'#' #'PR_CNN2D_fixed' #'PR_CNN2D'#'CNN_2D' BP_CNN2D_MULTIBP_PRFRTRAINABLEGAMMA

dataset_nameForPaths = ''
for i in range(len(dataset)):
    dataset_nameForPaths = dataset_nameForPaths+dataset[i]+'+'

dataset_nameForPaths = dataset_nameForPaths[:-1]

path_model_save_base = os.path.join(base,'analyses/data_'+data_pers+'/',expDate,subFold,'models',dataset_nameForPaths,mdl_subFold)
path_dataset_base = os.path.join('/home/saad/postdoc_db/analyses/data_'+data_pers+'/',expDate,subFold)

fname_data_train_val_test = ''
i=0
for i in range(len(dataset)):
    name_datasetFile = expDate+'_dataset_train_val_test_'+dataset[i]+'.h5'
    fname_data_train_val_test = fname_data_train_val_test+os.path.join(path_dataset_base,'datasets',name_datasetFile) + '+'
fname_data_train_val_test = fname_data_train_val_test[:-1]

fname_allParams = os.listdir(os.path.join(path_model_save_base,mdl_name))

recalculate_perf = 0
# %%
counter = 0
fname_param = fname_allParams[0]
for fname_param in fname_allParams:
    counter = counter+1
    print('%d of %d\n' %(counter,len(fname_allParams)))
    current_epoch = len([fname_param for fname_param in os.listdir(os.path.join(path_model_save_base,mdl_name,fname_param)) if fname_param.endswith('index')])
    
    fname_performance = os.path.join(path_model_save_base,mdl_name,fname_param,'performance',expDate+'_'+fname_param+'.h5')
    perf_exist = os.path.exists(fname_performance)
    try:
        perf = h5py.File(fname_performance,'r')
        nepochsAtPerfCalc = perf['model_performance']['fev_medianUnits_allEpochs'].shape[0]
        perf.close()
    except:
        nepochsAtPerfCalc = 0
    
    
    # if current_epoch>min_nepochs and not(recalculate_perf==0 and perf_exist):
    if current_epoch>min_nepochs and nepochsAtPerfCalc<min_nepochs:
            
        params = getModelParams(fname_param)
        
    
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
        chan4_n=0#params['C4_n']
        filt4_size=0#params['C4_s']
        filt4_3rdDim=0#params['C4_3d']

        trainingSamps_dur = 0 # minutes
        nb_epochs=0         # setting this to 0 only runs evaluation
        bz_ms=5000#20000 #10000
        USE_CHUNKER = 1
        BatchNorm=params['BN']
        MaxPool=params['MP']
        runOnCluster=0
        c_trial = params['TR']
        trainingSamps_dur = params['TRSAMPS']
        
        BatchNorm_train = 0
        saveToCSV=1
        # trainingSamps_dur=0
        validationSamps_dur=0
        testSamps_dur=0.1
        
        idx_unitsToTake = np.array([0])#idx_units_ON_train #[0] #idx_units_train
        select_rgctype=0
        CONTINUE_TRAINING = 1


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
                                trainingSamps_dur=trainingSamps_dur,validationSamps_dur=validationSamps_dur,testSamps_dur=testSamps_dur,idx_unitsToTake=idx_unitsToTake,
                                lr=lr,lr_fac=lr_fac,use_lrscheduler=use_lrscheduler)
        
        plt.plot(model_performance['fev_medianUnits_allEpochs']);plt.ylabel('FEV');plt.xlabel('Epochs');plt.show()
        print('FEV = %0.2f' %(np.nanmax(model_performance['fev_medianUnits_allEpochs'])*100))

        
        # model_performance,mdl = run_model(expDate,mdl_name,path_model_save_base,fname_data_train_val_test,path_dataset_base=path_dataset_base,saveToCSV=saveToCSV,runOnCluster=0,
        #                         temporal_width=temporal_width, pr_temporal_width=pr_temporal_width, thresh_rr=thresh_rr,
        #                         chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim,
        #                         chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim,
        #                         chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim,
        #                         nb_epochs=nb_epochs,bz_ms=bz_ms,
        #                         BatchNorm=BatchNorm,BatchNorm_train = BatchNorm_train,MaxPool=MaxPool,c_trial=c_trial,CONTINUE_TRAINING=1,info='',
        #                         trainingSamps_dur=trainingSamps_dur,validationSamps_dur=validationSamps_dur,lr=lr,USE_CHUNKER=USE_CHUNKER)
        
        # plt.plot(model_performance['fev_medianUnits_allEpochs'])
        # plt.title(f)
        # print('Model: %s\nFEV = %0.2f\n' %(f,(np.max(model_performance['fev_medianUnits_allEpochs'])*100)))
    
    
