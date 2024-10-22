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
from model import utils_si
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import socket

hostname=socket.gethostname() 
if hostname=='sandwolf':
    base = '/home/saad/data_hdd/'
elif hostname=='sandhound':
    base = '/home/saad/postdoc_db/'

# base = '/home/saad/postdoc_db/'
base = '/home/saad/data_hdd/'


from scipy.stats import wilcoxon
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

# from pyret.filtertools import sta, decompose
# import seaborn

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Input, Reshape
from model.train_model import chunker



# %%
data_pers = 'mike'
expDates = ('20230725C',)
subFold = '' #'PR_BP' #'8ms_clark' #'8ms_trainablePR' # test_coneParams
mdl_subFold = 'finetuning_crossval'
lightLevel_1 =  ('NATSTIM6_CORR_mesopic-Rstar_f4_8ms',) #CB_CORR_mesopic-Rstar_f4_8ms  NATSTIM3_CORR_mesopic-Rstar_f4_8ms C
natstims_all = ('NATSTIM0_CORR_mesopic-Rstar_f4_8ms','NATSTIM1_CORR_mesopic-Rstar_f4_8ms', 'NATSTIM2_CORR_mesopic-Rstar_f4_8ms', 'NATSTIM3_CORR_mesopic-Rstar_f4_8ms','NATSTIM4_CORR_mesopic-Rstar_f4_8ms',
                'NATSTIM5_CORR_mesopic-Rstar_f4_8ms','NATSTIM6_CORR_mesopic-Rstar_f4_8ms','NATSTIM7_CORR_mesopic-Rstar_f4_8ms','NATSTIM8_CORR_mesopic-Rstar_f4_8ms')

lightLevel_2 = () #list(set(natstims_all)-set(lightLevel_1))#('CB_CORR_mesopic-Rstar_f4_8ms',)#('NATSTIM3_CORR_photopic-Rstar_f4',) CB_CORR_photopic-Rstar_f4_8ms
models_all = ('CNN2D',) # 
pr_params_name = ''

writeToCSV = False

path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance_mike'

models_list = []

param_list_keys = ['U', 'T','C1_n','C1_s','C1_3d','C2_n','C2_s','C2_3d','C3_n','C3_s','C3_3d','BN','MP','TR','P','LR','TRSAMPS']   # model parameters to group by
csv_header = ['mdl_name','expDate','temp_window','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim',
              'FEV_median','FracExVar','corr_median','rr_corr_median']

cols_lightLevels = {
    'photopic': 'r',
    'scotopic': 'b',
    'scot-0.3':'b',
    'scot-0.3-R':'b',
    'scot-3': 'r',
    'scot-3-R': 'r',
    'scot-30':'g',
    'scot-30-':'g'
    }

# Dataset info
dataset_nameForPaths = ''
for i in range(len(lightLevel_1)):
    dataset_nameForPaths = dataset_nameForPaths+lightLevel_1[i]+'+'

dataset_nameForPaths = dataset_nameForPaths[:-1]

path_dataset_base = os.path.join('/home/saad/postdoc_db/analyses/data_'+data_pers+'/',expDates[0],subFold)
path_dataset = os.path.join(path_dataset_base,'datasets')

fname_data_train_val_test = ''
i=0
for i in range(len(lightLevel_1)):
    name_datasetFile = expDates[0]+'_dataset_train_val_test_'+lightLevel_1[i]+'.h5'
    fname_data_train_val_test = fname_data_train_val_test+os.path.join(path_dataset_base,'datasets',name_datasetFile) + '+'
fname_data_train_val_test = fname_data_train_val_test[:-1]

fname_data_train_val_test_2 = ''
i=0
for i in range(len(lightLevel_2)):
    name_datasetFile = expDates[0]+'_dataset_train_val_test_'+lightLevel_2[i]+'.h5'
    fname_data_train_val_test_2 = fname_data_train_val_test_2+os.path.join(path_dataset_base,'datasets',name_datasetFile) + '+'
fname_data_train_val_test_2 = fname_data_train_val_test_2[:-1]


perf_allExps = {}   # Performance dict for all experiments, models, conditions
params_allExps = {} # Dict containing parameter values for all experiments, models, conditions
paramNames_allExps = {} #Dict containing the folder name of the model, which is basically concatenating the params together

idx_exp = 0
select_mdl = models_all[0]

for idx_exp in range(len(expDates)):
    
    path_mdl_base = os.path.join(base,'analyses/data_'+data_pers+'/',expDates[idx_exp],'models',subFold)
    path_dataset_base = os.path.join('/home/saad/postdoc_db/analyses/data_'+data_pers+'/',expDates[idx_exp],subFold)

    perf_allModels = {}
    params_allModels = {}
    paramNames_allModels = {}
    exp_select = expDates[idx_exp]
    
    path_mdl_drive = os.path.join(path_mdl_base,dataset_nameForPaths,mdl_subFold)

    
    # select_mdl = models_all[0]
    for select_mdl in models_all:

        if writeToCSV==True:
            fname_csv_file = 'performance_'+exp_select+'_'+lightLevel_1+'_avgAcrossTrials_'+select_mdl+'.csv'
            fname_csv_file = os.path.join(path_save_performance,fname_csv_file)
            with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 

        # Get the folder names (paramNames) for the different runs of select_mdl
        path_mdl = os.path.join(path_mdl_drive,select_mdl)
        if 'PRFR' in select_mdl or pr_params_name=='0':
            path_mdl = os.path.join(path_mdl,pr_params_name)

        paramNames_temp = os.listdir(path_mdl)
        paramNames = ([])
        for p in paramNames_temp:
            try:     # some model runs are incomplete so don't consider them. They will not contain the performance file
                if os.listdir(os.path.join(path_mdl,p,'performance')):
                    paramNames.append(p)
            except:
                pass
        
        # paramNames = paramNames_temp
        
        # Cut out the trial number to group different trials of same models/params together
        paramNames_cut = []
        for i in range(0,len(paramNames)):
            rgb = re.compile(r'_TR-(\d+)')
            idx_paramMention = rgb.search(paramNames[i]).span()
            paramNames_cut_rgb = paramNames[i][:idx_paramMention[0]] + paramNames[i][idx_paramMention[1]:]
            paramNames_cut.append(paramNames_cut_rgb)
        paramNames_unique = list(set(paramNames_cut))   # This contains the unique runs
        
        # Locate where the unique names lie in the non-unique array. So this will also tell how many trials for each model/param
        idx_paramName = ([])
        for i in range(len(paramNames_unique)):
            name = paramNames_unique[i]
            rgb = [j for j,k in enumerate(paramNames_cut) if name==k]
            idx_paramName.append(rgb)
            
    
        perf_paramNames = {}
        param_list = dict([(key, []) for key in param_list_keys])


        # param_unique_idx = 0
        for param_unique_idx in range(len(paramNames_unique)):
            
            # initialize variables where the last dimension size is number of trials for that model/params combo
            num_trials = len(idx_paramName[param_unique_idx])
            
            fev_allUnits_allEpochs_allTr = np.zeros((1000,1000,num_trials))           
            fev_allUnits_bestEpoch_allTr = np.zeros((1000,num_trials))
            fev_medianUnits_allEpochs_allTr = np.zeros((1000,num_trials))
            fev_medianUnits_bestEpoch_allTr = np.zeros((num_trials))

            predCorr_allUnits_allEpochs_allTr = np.zeros((1000,1000,num_trials))           
            predCorr_allUnits_bestEpoch_allTr = np.zeros((1000,num_trials))
            predCorr_medianUnits_allEpochs_allTr = np.zeros((1000,num_trials))
            predCorr_medianUnits_bestEpoch_allTr = np.zeros((num_trials))
            
            trial_id = np.zeros((num_trials))
            
            
            for c_tr in range(num_trials):
            
                paramFileName = paramNames[idx_paramName[param_unique_idx][c_tr]]
                paramName_path = os.path.join(path_mdl,paramFileName)
                fname_performanceFile = os.path.join(paramName_path,'performance',exp_select+'_'+paramFileName+'.h5')
        
                f = h5py.File(fname_performanceFile,'r')
                
                rgb = np.atleast_1d(f['model_performance']['fev_allUnits_allEpochs'])
                num_units = rgb.shape[1]
                num_epochs = rgb.shape[0]
                fev_allUnits_allEpochs_allTr[:num_epochs,:num_units,c_tr] = rgb            

                rgb = np.array(f['model_performance']['fev_allUnits_bestEpoch'])
                fev_allUnits_bestEpoch_allTr[:rgb.shape[0],c_tr] = rgb
                try:
                    rgb = np.array(f['model_performance']['fev_median_allEpochs'])
                except:
                    rgb = np.array(f['model_performance']['fev_medianUnits_allEpochs'])
                rgb = np.array(f['model_performance']['fev_medianUnits_allEpochs'])
                fev_medianUnits_allEpochs_allTr[:rgb.shape[0],c_tr] = rgb
                rgb = np.atleast_1d(f['model_performance']['fev_medianUnits_bestEpoch'])
                fev_medianUnits_bestEpoch_allTr[c_tr] = rgb
                
                rgb = np.array(f['model_performance']['predCorr_allUnits_bestEpoch'])
                predCorr_allUnits_bestEpoch_allTr[:rgb.shape[0],c_tr] = rgb
                rgb = np.array(f['model_performance']['predCorr_medianUnits_allEpochs'])
                predCorr_medianUnits_allEpochs_allTr[:rgb.shape[0],c_tr] = rgb
                rgb = np.atleast_1d(f['model_performance']['predCorr_medianUnits_bestEpoch'])
                predCorr_medianUnits_bestEpoch_allTr[c_tr] = rgb
                
                rgb = getModelParams(os.path.split(fname_performanceFile)[-1])
                trial_id[c_tr] = np.atleast_1d(rgb['TR'])
                
            
            
            fev_allUnits_allEpochs_allTr = fev_allUnits_allEpochs_allTr[:num_epochs,:num_units,:]
            fev_allUnits_bestEpoch_allTr = fev_allUnits_bestEpoch_allTr[:num_units,:]
            fev_medianUnits_allEpochs_allTr = fev_medianUnits_allEpochs_allTr[:num_epochs,:]
            fev_medianUnits_bestEpoch_allTr = fev_medianUnits_bestEpoch_allTr
            fracExVar_allUnits = np.array(f['model_performance']['fracExVar_allUnits'])
            fracExVar_medianUnits = np.array(f['model_performance']['fracExVar_medianUnits'])
            
            
            predCorr_allUnits_allEpochs_allTr = fev_allUnits_allEpochs_allTr[:num_epochs,:num_units,:]
            predCorr_allUnits_bestEpoch_allTr = predCorr_allUnits_bestEpoch_allTr[:num_units,:]
            predCorr_medianUnits_allEpochs_allTr = fev_medianUnits_allEpochs_allTr[:num_epochs,:]
            predCorr_medianUnits_bestEpoch_allTr = predCorr_medianUnits_bestEpoch_allTr
            rrCorr_allUnits =  np.array(f['model_performance']['rrCorr_allUnits'])
            rrCorr_medianUnits = np.array(f['model_performance']['rrCorr_medianUnits'])

            
            perf = {
                'fev_allUnits_allEpochs_allTr': fev_allUnits_allEpochs_allTr,
                'fev_allUnits_bestEpoch_allTr': fev_allUnits_bestEpoch_allTr,
                'fev_medianUnits_allEpochs_allTr': fev_medianUnits_allEpochs_allTr,
                'fev_medianUnits_bestEpoch_allTr': fev_medianUnits_bestEpoch_allTr,
                'fracExVar_allUnits':  fracExVar_allUnits,
                'fracExVar_medianUnits':  fracExVar_medianUnits,    
                'idx_bestEpoch': np.array(f['model_performance']['idx_bestEpoch']),
                
                
                'predCorr_allUnits_allEpochs_allTr': predCorr_allUnits_allEpochs_allTr,
                'predCorr_allUnits_bestEpoch_allTr': predCorr_allUnits_bestEpoch_allTr,
                'predCorr_medianUnits_allEpochs_allTr': predCorr_medianUnits_allEpochs_allTr,
                'predCorr_medianUnits_bestEpoch_allTr': predCorr_medianUnits_bestEpoch_allTr,
                'rrCorr_allUnits':  rrCorr_allUnits,
                'rrCorr_medianUnits':  rrCorr_medianUnits,                
                'num_trials': num_trials,
                # 'val_dataset_name': utils_si.h5_tostring(np.array(f['model_performance']['val_dataset_name'])),
                'trial_id': trial_id,
                }
            
            performance = {}
            performance['model_performance'] = perf

            select_groups = ('model_params','data_quality')
            

            for j in select_groups:
                perf_group = {}
                
                keys = list(f[j].keys())
                
                for i in keys:
                    rgb = np.array(f[j][i])
                    rgb_type = rgb.dtype.name
                       
                    if 'bytes' in rgb_type:
                        perf_group[i] = utils_si.h5_tostring(rgb)
                    else:
                        perf_group[i] = rgb
                        
                performance[j] = perf_group
                    
            # filt_temporal_width = np.array(f['stim_info']['temporal_width'])
            filt_temporal_width = 120
            performance['model_params']['thresh_rr'] = np.array(f['thresh_rr'])
            performance['model_params']['idx_unitsToTake'] = np.array(f['idx_unitsToTake'])
            performance['model_params']['temporal_width'] = filt_temporal_width
            
            pr_param_list = ('pr_alpha','pr_beta','pr_gamma','pr_nY','pr_nZ','pr_tauY','pr_tauZ')
            perf_keys = list(f['model_performance'].keys())
            if 'pr_alpha' in perf_keys:
                for j in pr_param_list:
                    performance['model_params'][j] = np.array(f['model_performance'][j])
            
            
            rgb = getModelParams(paramFileName)
            for i in list(rgb.keys()):
                param_list[i].append(rgb[i])
            
            
            perf_paramNames[paramNames_unique[param_unique_idx]] = performance
            
            
            
            if writeToCSV == True:
                csv_data = [select_mdl,exp_select,filt_temporal_width,rgb['C1_n'], rgb['C1_s'], rgb['C1_3d'], rgb['C2_n'], rgb['C2_s'], rgb['C2_3d'], rgb['C3_n'], rgb['C3_s'], rgb['C3_3d'],np.nanmean(fev_medianUnits_bestEpoch_allTr),fracExVar_medianUnits,np.mean(predCorr_medianUnits_bestEpoch_allTr),rrCorr_medianUnits]
                with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
                        csvwriter = csv.writer(csvfile) 
                        csvwriter.writerow(csv_data) 
                        
                        
            f.close()

            
            # p_U
            
        perf_allModels[select_mdl] = perf_paramNames
        params_allModels[select_mdl] = param_list
        paramNames_allModels[select_mdl] = paramNames_unique

        
    perf_allExps[exp_select] = perf_allModels       
    params_allExps[exp_select] = params_allModels
    paramNames_allExps[exp_select] = paramNames_allModels

    
# del perf_allModels, perf_paramNames, performance, perf_group, param_list, params_allModels, paramNames_allModels, paramNames


# %% D1: Test model
select_exp = expDates[0]
select_mdl = models_all[0] #'CNN_2D' #'CNN_2D_chansVary'#'CNN_2D_filtsVary'
# params_mdl = params_allExps[select_exp][select_mdl]

val_dataset_1 = lightLevel_1 #'scot-3-Rstar_mdl-rieke_s-7.07_p-7.07_e-2.53_k-0.01_h-3_b-25_hc-4_gd-15.5_g-50.0_preproc-rods_norm-0_tb-8_Euler_RF-2' #lightLevel_1  #lightLevel_1 #      # ['scotopic','photopic']
correctMedian = False

select_LR = '0.0001'
select_U = 57
select_P = 0
select_T = 80
select_BN = 1
select_MP = 2
# select_TR = 1
select_CB = 0
select_C1_n = 10
select_C1_s = 11#9
select_C1_3d = 0
select_C2_n = 15
select_C2_s = 7
select_C2_3d = 0
select_C3_n = 20
select_C3_s = 7
select_C3_3d = 0
select_TRSAMPS = -1

paramFileName,_ = modelFileName(U=select_U,P=select_P,T=select_T,BN=select_BN,MP=select_MP,LR=select_LR,CB_n=select_CB,
                 C1_n=select_C1_n,C1_s=select_C1_s,C1_3d=select_C1_3d,
                 C2_n=select_C2_n,C2_s=select_C2_s,C2_3d=select_C2_3d,
                 C3_n=select_C3_n,C3_s=select_C3_s,C3_3d=select_C3_3d,with_TR=False,
                 TRSAMPS=select_TRSAMPS,with_TRSAMPS=True)

# val_dataset_1 = perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['val_dataset_name'][0]


# assert val_dataset_1 != val_dataset_1, 'same datasets selected'
idx_bestTrial = np.nanargmax(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
# idx_bestEpoch = np.nanargmax(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])
idx_bestEpoch = len(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])-1
trial_num = perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['trial_id'][idx_bestTrial]
plt.plot(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])
select_TR = int(trial_num)

unames = np.array(perf_allExps[select_exp][select_mdl][paramFileName]['data_quality']['uname_selectedUnits'])


mdlFolder = paramFileName+'_TR-%02d' % select_TR
path_model = os.path.join(path_mdl_drive,select_mdl)
if 'PRFR' in select_mdl and pr_params_name !='':
    path_model = os.path.join(path_model,pr_params_name)
else:
    try:
        tf.compat.v1.disable_eager_execution()
    except:
        pass


path_model = os.path.join(path_model,mdlFolder)
mdl = load(os.path.join(path_model,mdlFolder))

if idx_bestEpoch !=-1:
    fname_bestWeight = 'weights_'+mdlFolder+'_epoch-%03d' % (idx_bestEpoch+1)
else:
    mdl_epochList = np.sort(glob.glob(os.path.join(path_model,'*.index')))
    lastEpoch = mdl_epochList[-1].split('.index')[0]
    fname_bestWeight = os.path.split(lastEpoch)[-1]

try:
    mdl.load_weights(os.path.join(path_model,fname_bestWeight))
except:
    mdl.load_weights(os.path.join(path_model,fname_bestWeight+'.h5'))
weights_dict = get_weightsDict(mdl)



fname_data_train_val_test_all = fname_data_train_val_test.split('+')
if len(fname_data_train_val_test_2)>0:
    fname_data_train_val_test_all=fname_data_train_val_test_all+fname_data_train_val_test_2.split('+')
# fname_data_train_val_test_all = ['']


idx_train_start = 0    # mins to chop off in the begining.
d=0
dict_test = {}
dict_val = {}
dict_train = {}
for d in range(len(fname_data_train_val_test_all)):
    rgb = load_h5Dataset(fname_data_train_val_test_all[d],nsamps_val=0.5,nsamps_test=0.1,nsamps_train=0.05,LOAD_TR=False)
    data_train=rgb[0];
    data_val = rgb[1];
    data_test = rgb[2];
    data_quality = rgb[3]
    dataset_rr = rgb[4]
    parameters = rgb[5]
    resp_orig = rgb[6]
    if len(rgb)>7:
        data_info = rgb[7]

    t_frame = parameters['t_frame']     # time in ms of one frame/sample 

    if select_mdl[:6]=='CNN_2D' or select_mdl == 'replaceDense_2D' or select_mdl[:5]=='CNN2D':
        # data_train = prepare_data_cnn2d(data_train,select_T,np.arange(data_train.y.shape[1]))
        data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
        data_test = prepare_data_cnn2d(data_test,select_T,np.arange(data_test.y.shape[1]))
    elif select_mdl[:6]=='CNN_3D':
        data_val = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))
        # data_test = prepare_data_cnn3d(data_test,select_T,np.arange(data_test.y.shape[1]))
        
    elif 'PR' in select_mdl:
        pr_temporal_width = select_P
        data_val = prepare_data_cnn2d(data_val,pr_temporal_width,np.arange(data_val.y.shape[1]))
        data_test = prepare_data_cnn2d(data_test,pr_temporal_width,np.arange(data_test.y.shape[1]))

    
    elif select_mdl[:8]=='PR_CNN3D':
        pr_temporal_width = perf_allExps[select_exp][select_mdl][paramFileName]['model_params']['pr_temporal_width']
        data_val = prepare_data_cnn3d(data_val,pr_temporal_width,np.arange(data_val.y.shape[1]))
        obs_rate_allStimTrials_d1 = dattaset_rr['stim_0']['val'][:,pr_temporal_width:,:]

    dict_train[fname_data_train_val_test_all[d]] = data_train
    dict_val[fname_data_train_val_test_all[d]] = data_val
    dict_test[fname_data_train_val_test_all[d]] = data_test

# data_train = model.data_handler.merge_datasets(dict_train)
data_val = model.data_handler.merge_datasets(dict_val)
data_test = model.data_handler.merge_datasets(dict_test)

"""
from skimage import exposure
import numpy as np
def histogram_equalize(img):
    # img = rgb2gray(img)
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)

X = data_val.X.copy()
X = histogram_equalize(X) #(X-X.min())/(X.max()-X.min())
data_val = Exptdata(X,data_val.y)
"""
# if isinstance(data_train.X,list):
#     if len(data_train.X)>1000:
#         X = data_train.X[:1000]
#         X = np.asarray(X)
#         y = data_train.y[:1000]
#         y = np.asarray(y)
#         data_train = Exptdata(X,y)

if 'var_noise' in data_quality:
    obs_noise = data_quality['var_noise'] #estimate_noise(data_test.y_trials)
else:
    obs_noise = 0

"""
data_val_orig = data_val

a = np.moveaxis(data_val.y_trials,0,-1)
a = a.reshape(a.shape[0],-1)
a = np.moveaxis(a,-1,0)

plt.plot(a);plt.plot(data_val.y,'k');plt.show()
plt.hist(a.flatten())
plt.hist(data_val.y.flatten())
plt.hist(data_train.y.flatten())
"""

val_select = 'data_val'
# val_select = 'data_train'

data_val_select = eval(val_select)

print('Validation dataset: %s'%val_select)

dict_dsetPerf = {}
if isintuple(data_val_select,'dset_names'):
    unique_dsets = np.unique(data_val_select.dset_names)
    dset_name = unique_dsets[0]
    for dset_name in unique_dsets:
        idx = [i for i,n in enumerate(data_val_select.dset_names) if re.search(dset_name,n)]
        dict_dsetPerf[dset_name] = dict(idx=idx)


# idx_unitsToTake = perf_allExps[select_exp][select_mdl][paramFileName]['model_params']['idx_unitsToTake']
idx_unitsToTake = np.arange(0,57)

if data_val_select.y.ndim==3:
    # obs_rate_allStimTrials_d1 = np.moveaxis(data_val_select.y,-1,0)
    obs_rate_allStimTrials_d1 = obs_rate_allStimTrials_d1[:,:,idx_unitsToTake]
else:
    obs_rate_allStimTrials_d1 = data_val_select.y[:,idx_unitsToTake]
    # obs_rate_allStimTrials_d1 = np.moveaxis(np.squeeze(data_val_select.y_trials[:,idx_unitsToTake]),-1,0)

# obs_rate = obs_rate_allStimTrials_d1.copy()
obs_rate = data_val_select.y[:,idx_unitsToTake]

if correctMedian==True:
    fname_data_train_val_test_d1 = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_1+'.h5'))
    _,_,_,_,_,_,resp_med_d1 = load_h5Dataset(fname_data_train_val_test_d1)
    resp_med_d1 = np.nanmedian(resp_med_d1['train'],axis=0)
    resp_med_d2 = np.nanmedian(resp_orig,axis=0)
    resp_mulFac = resp_med_d2/resp_med_d1;
    
    pred_rate = mdl.predict(data_val_select.X)
    pred_rate = pred_rate * resp_mulFac[None,:]

    
else:
    batch_size = 32
    gen = chunker(data_val_select,batch_size=batch_size)    
    steps_per_epoch = int(np.ceil(len(data_val_select.X)/batch_size))
    pred_rate = mdl.predict_generator(gen,steps=steps_per_epoch)
    # pred_rate = mdl.predict(data_val_select.X,batch_size = 16)

"""
pred_rate = pred_rate
obs_rate = np.asarray(data_val_select.y)
"""
# obs_rate = data_test.y
# pred_rate = mdl.predict(data_test.X)
# 
samps_shift = 0#parameters['samps_shift']
num_iters = 1
fev_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d1_allUnits[:,i], fracExplainableVar[:,i], predCorr_d1_allUnits[:,i], rrCorr_d1_allUnits[:,i] = model_evaluate_new(
        obs_rate_allStimTrials_d1.copy(),pred_rate.copy(),0,RR_ONLY=False,lag = samps_shift,obs_noise=obs_noise)


fev_d1_allUnits = np.mean(fev_d1_allUnits,axis=1)
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d1_allUnits = np.mean(predCorr_d1_allUnits,axis=1)
rrCorr_d1_allUnits = np.mean(rrCorr_d1_allUnits,axis=1)

idx_allUnits = np.arange(fev_d1_allUnits.shape[0])
idx_d1_valid = idx_allUnits
# idx_d1_valid = fev_d1_allUnits<1.1
# idx_d1_valid = idx_allUnits[idx_d1_valid]
print('N = %d RGCs'%len(idx_d1_valid))

# fev_d1_validUnits = fev_d1_allUnits[idx_d1_valid]
# predCorr_d1_validUnits = predCorr_d1_allUnits[idx_d1_valid]
# rrCorr_d1_validUnits = rrCorr_d1_allUnits[idx_d1_valid]
# # fev_d1_allUnits = fev_d1_allUnits[idx_valid]

fev_d1_medianUnits = np.nanmedian(fev_d1_allUnits[idx_d1_valid])
print('FEV = %0.2f' %(fev_d1_medianUnits*100))
fev_d1_stdUnits = np.nanstd(fev_d1_allUnits[idx_d1_valid])
fev_d1_ci = 1.96*(fev_d1_stdUnits/len(idx_d1_valid)**.5)

predCorr_d1_medianUnits = np.nanmedian(predCorr_d1_allUnits[idx_d1_valid])
print('R = %0.2f' %predCorr_d1_medianUnits)
predCorr_d1_stdUnits = np.nanstd(predCorr_d1_allUnits[idx_d1_valid])
predCorr_d1_ci = 1.96*(predCorr_d1_stdUnits/len(idx_d1_valid)**.5)

rrCorr_d1_medianUnits = np.median(rrCorr_d1_allUnits[idx_d1_valid])
rrCorr_d1_stdUnits = np.std(rrCorr_d1_allUnits[idx_d1_valid])
rrCorr_d1_ci = 1.96*(rrCorr_d1_stdUnits/len(idx_d1_valid)**.5)


idx_units_sorted = np.argsort(fev_d1_allUnits[idx_d1_valid])
idx_units_sorted = idx_d1_valid[idx_units_sorted]
# idx_unitsToPred = [idx_units_sorted[0],idx_units_sorted[1],idx_units_sorted[2],idx_units_sorted[3]]
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[-3],idx_units_sorted[-4]]
# %%
idx_unitsToPred = [2,52,17,47]     # Unit 22 is good for natstim comparisons
uname_sorted = unames[idx_units_sorted]
idx_on = np.intersect1d(np.arange(0,27),idx_d1_valid)
idx_off = np.intersect1d(np.arange(27,57),idx_d1_valid)

fev_median_on = np.nanmedian(fev_d1_allUnits[idx_on])
fev_median_off = np.nanmedian(fev_d1_allUnits[idx_off])
print('FEV_ON = %0.2f' %(fev_median_on*100))
print('FEV_OFF = %0.2f' %(fev_median_off*100))

"""
# Plot images
fig,axs=plt.subplots(3,3,figsize=(10,10))
axs=np.ravel(axs)
i=-1
for key in dict_dsetPerf.keys():
    i=i+1
    idx = np.asarray(dict_dsetPerf[key]['idx'])
    img = data_val.X[idx][-1,-1,:,:]
    bins = np.arange(0,2.05,0.05)
    # axs[i].imshow(img,cmap='gray')
    axs[i].hist(img.flatten(),bins)
    axs[i].set_title(key)
    
fig,axs=plt.subplots(3,3,figsize=(15,15))
axs=np.ravel(axs)
i=-1
for key in dict_dsetPerf.keys():
    i=i+1
    idx = np.asarray(dict_dsetPerf[key]['idx'])
    img = data_val.X[idx][:,-1,:,:]
    bins = np.arange(0,2.05,0.05)
    axs[i].hist(img.flatten(),bins)
    axs[i].set_title(key)

"""

if len(dict_dsetPerf)>0:
    for key in dict_dsetPerf.keys():
        idx = np.asarray(dict_dsetPerf[key]['idx'])
        rgb_fev_d1_allUnits,_, rgb_predCorr_d1_allUnits, _ = model_evaluate_new(
            obs_rate_allStimTrials_d1[idx,:].copy(),pred_rate[idx,:].copy(),0,RR_ONLY=False,lag=samps_shift,obs_noise=obs_noise)
        dict_dsetPerf[key]['fev_d1_allUnits'] = rgb_fev_d1_allUnits
        dict_dsetPerf[key]['fev_d1_medianUnits'] = np.nanmedian(rgb_fev_d1_allUnits[idx_d1_valid])
        dict_dsetPerf[key]['predCorr_d1_allUnits'] = rgb_predCorr_d1_allUnits
        dict_dsetPerf[key]['predCorr_d1_medianUnits'] = np.nanmedian(rgb_predCorr_d1_allUnits[idx_d1_valid])
        
        dict_dsetPerf[key]['fev_d1_median_ON'] = np.nanmedian(rgb_fev_d1_allUnits[idx_on])
        dict_dsetPerf[key]['fev_d1_median_OFF'] = np.nanmedian(rgb_fev_d1_allUnits[idx_off])

        print('%s: FEV = %0.2f'%(key,dict_dsetPerf[key]['fev_d1_medianUnits']*100))
        print('%s: R = %0.2f'%(key,dict_dsetPerf[key]['predCorr_d1_medianUnits']))
        # print('%s: FEV_ON = %0.2f'%(key,dict_dsetPerf[key]['fev_d1_median_ON']*100))
        # print('%s: FEV_OFF = %0.2f'%(key,dict_dsetPerf[key]['fev_d1_median_OFF']*100))



t_start = 0
t_dur = obs_rate.shape[0]
t_end = t_dur
win_display = (t_start,t_start+t_dur)
font_size_ticks = 12
font_size_leg = 16

t_frame = 8
t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

col_mdl = ('b')
lineWidth_mdl = [2]
lim_y = (0,6)
fig,axs = plt.subplots(2,2,figsize=(25,10))
axs = np.ravel(axs)
# fig.suptitle('Training: '+lightLevel_1+' | Prediction: '+ val_dataset_1,fontsize=16)
for i in range(len(idx_unitsToPred)):
    l_base, = axs[i].plot(t_axis[t_start+samps_shift:t_end],obs_rate[t_start:t_end-samps_shift,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+samps_shift:t_end],pred_rate[t_start+samps_shift:t_end,idx_unitsToPred[i]],linewidth=lineWidth_mdl[0])
    l.set_label('Predicted: FEV = %.02f, Corr = %0.2f' %(fev_d1_allUnits[idx_unitsToPred[i]],predCorr_d1_allUnits[idx_unitsToPred[i]])) 
    
    # axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate (spikes/second)',fontsize=font_size_ticks)
    axs[i].set_title('unit_id: %d' %idx_unitsToPred[i])
    axs[i].legend(fontsize=font_size_leg,loc='best')
    plt.setp(axs[i].get_xticklabels(), fontsize=font_size_ticks)


# %%

"""
idx_valid = fev_d1_allUnits<1.1
idx_valid = idx_allUnits[idx_valid]
fev_pr_fixed = fev_d1_allUnits[idx_valid].copy()*100
fev_pr_fixed_med = np.median(fev_pr_fixed)
fev_std_pr_fixed = np.nanstd(fev_pr_fixed)
fev_ci_pr_fixed = 1.96*(fev_std_pr_fixed/len(idx_valid)**.5)

idx_valid = idx_allUnits[idx_valid]
fev_pr_trainable = fev_d1_allUnits[idx_valid].copy()*100
fev_pr_trainable_med = np.median(fev_pr_trainable)
fev_std_pr_trainable = np.nanstd(fev_pr_trainable)
fev_ci_pr_trainable = 1.96*(fev_std_pr_trainable/len(idx_valid)**.5)

idx_valid = idx_allUnits[idx_valid]
fev_pr_ln = fev_d1_allUnits[idx_valid].copy()*100
fev_pr_ln_med = np.median(fev_pr_ln)
fev_std_pr_ln = np.nanstd(fev_pr_ln)
fev_ci_pr_ln = 1.96*(fev_std_pr_ln/len(idx_valid)**.5)


fev_cnn = fev_d1_allUnits[idx_valid].copy()*100
fev_cnn_med = np.median(fev_cnn)
fev_std_cnn = np.nanstd(fev_cnn)
fev_ci_cnn = 1.96*(fev_std_cnn/len(idx_valid)**.5)


--------

idx_valid = dict_dsetPerf['NATSTIM3_CORR_mesopic']['fev_d1_allUnits']<1.1
idx_valid = idx_allUnits[idx_valid]
fev_pr = dict_dsetPerf['NATSTIM3_CORR_mesopic']['fev_d1_allUnits'][idx_valid].copy()*100
fev_pr_med = np.median(fev_pr)
fev_std_pr = np.nanstd(fev_pr)
fev_ci_pr = 1.96*(fev_std_pr/len(idx_valid)**.5)


fev_cnn = dict_dsetPerf['NATSTIM3_CORR_mesopic']['fev_d1_allUnits'][idx_valid].copy()*100
fev_cnn_med = np.median(fev_cnn)
fev_std_cnn = np.nanstd(fev_cnn)
fev_ci_cnn = 1.96*(fev_std_cnn/len(idx_valid)**.5)



"""
# import seaborn as sns

fevs = [fev_cnn,fev_pr_fixed,fev_pr_trainable,fev_pr_ln]
fevs_med = np.median(fevs,axis=1)
# print('median FEVs: CNN = %0.2f | PR-CNN = %0.2f'%(np.median(fev_cnn),np.median(fev_pr)))
cis = [fev_ci_cnn,fev_ci_pr_fixed,fev_ci_pr_trainable,fev_ci_pr_ln]
fig,axs = plt.subplots(1,1,figsize=(12,8))
axs = np.ravel(axs)
fig.suptitle('Training: '+lightLevel_1[0],fontsize=16)
font_size_ticks = 20

col_scheme = ('blue','green')
xlabel = ['CNN','fixedPR_CNN','trainablePR_CNN','linearPR_CNN']
axs[0].boxplot(fevs)
axs[0].set_xticklabels(xlabel)
axs[0].yaxis.grid(True)

# axs[0].boxplot(xlabel,fevs,yerr=cis,align='center',capsize=6,alpha=.7,color=[col_scheme],width=0.4)
axs[0].set_ylabel('Median FEV (%) | N = 57 RGCs',fontsize=font_size_ticks)
axs[0].set_title('',fontsize=font_size_ticks)
axs[0].set_yticks(np.arange(-40,110,10))
axs[0].set_ylim((0,110))

# axs[0].text(-0.2,1,'FEV = %0.2f' %fevs[0],fontsize=font_size_ticks)
axs[0].tick_params(axis='both',labelsize=font_size_ticks)
# _ = gc.collect()

# %% 

# #Temporal width 60
# tr = np.array([10,15,20,25,30,35])
# fev_cnn = np.array([52,60,61,64,64,64])
# fev_rods_fixed = np.array([55,55,66,70,74,72])
# fev_rods_trainable = np.array([53,65,65,74,76,76])
# fev_pr_ln = np.array([49,61,60,62,66,66])
# # fev_cones_trainable = np.array([53,61,65,67,71,74])
# # fev_cones_fixed = np.array([44,50,55,58,63,65])

# fig,axs = plt.subplots(1,1,figsize=(5,5))
# axs = np.ravel(axs)
# font_size_ticks = 20

# # xlabel = ['cones_fixed','rods_fixed']
# axs[0].plot(tr,fev_cnn,'-ko',label='cnn')
# # axs[0].plot(tr,fev_cones_trainable,'-ro',label='cones_train')
# # axs[0].plot(tr,fev_cones_fixed,'--rx',label='cones_fixed')
# axs[0].plot(tr,fev_rods_trainable,'-go',label='rods_train')
# axs[0].plot(tr,fev_rods_fixed,'--gx',label='rods_fixed')
# axs[0].plot(tr,fev_pr_ln,'-o',label='linear PR')
# axs[0].set_xlabel('Training data (minutes)')
# axs[0].set_ylabel('Median FEV (%) | N = 57 RGCs')
# axs[0].legend(loc='best')


# # Temporal width 120
# tr = np.array([10,15,20,25,30,35])
# fev_cnn = np.array([61,67,70,73,73,76])
# # fev_cones_trainable = np.array([53,70,68,77,78,77])
# # fev_cones_fixed = np.array([56,68,67,73,72,74])
# fev_rods_fixed = np.array([55,65,67,73,76,74])
# fev_rods_trainable = np.array([57,66,71,76,75,76])
# fev_pr_ln = np.array([57,66,66,73,75,76])

# fig,axs = plt.subplots(1,1,figsize=(5,5))
# axs = np.ravel(axs)
# font_size_ticks = 20

# # xlabel = ['cones_fixed','rods_fixed']
# axs[0].plot(tr,fev_cnn,'-ko',label='cnn')
# # axs[0].plot(tr,fev_cones_trainable,'-ro',label='cones_train')
# # axs[0].plot(tr,fev_cones_fixed,'--rx',label='cones_fixed')
# axs[0].plot(tr,fev_rods_trainable,'-go',label='rods_train')
# axs[0].plot(tr,fev_rods_fixed,'--gx',label='rods_fixed')
# axs[0].plot(tr,fev_pr_ln,'-o',label='linear PR')
# axs[0].set_xlabel('Training data (minutes)')
# axs[0].set_ylabel('Median FEV (%) | N = 57 RGCs')
# axs[0].legend(loc='best')
# axs[0].set_title('120 ms')


# Temporal width 80 ms - 60
tr = np.array([10,15,20,25,30,35])
fev_cnn = np.array([61,67,68,71,72,72])
# fev_rods_fixed = np.array([55,55,66,70,74,72])
fev_rods_fixed = np.array([47,54,63,69,72,75])
fev_rods_trainable = np.array([53,65,65,74,76,74])
fev_pr_ln = np.array([49,61,60,62,66,66])

fig,axs = plt.subplots(1,1,figsize=(5,5))
axs = np.ravel(axs)
font_size_ticks = 20

# xlabel = ['cones_fixed','rods_fixed']
axs[0].plot(tr,fev_cnn,'-ko',label='cnn')
axs[0].plot(tr,fev_rods_trainable,'-go',label='rods_train')
axs[0].plot(tr,fev_rods_fixed,'--gx',label='rods_fixed')
# axs[0].plot(tr,fev_pr_ln,'-o',label='linear PR')
axs[0].set_xlabel('Training data (minutes)')
axs[0].set_ylabel('Median FEV (%) | N = 57 RGCs')
axs[0].legend(loc='best')
axs[0].set_title('MESOPIC 80 - 60 ms')


# Temporal width 80 ms - 60
tr = np.array([10,15,20,25,30,35,40,45,50,55])
fev_cnn = np.array([17,22,24,18,30,28,30,30,35,33])
# fev_rods_fixed = np.array([55,55,66,70,74,72])
# fev_rods_fixed = np.array([47,54,63,69,72,75])
fev_cones_trainable = np.array([13,30,20,29,36,37,34,34,36,40])
# fev_pr_ln = np.array([49,61,60,62,66,66])

fig,axs = plt.subplots(1,1,figsize=(5,5))
axs = np.ravel(axs)
font_size_ticks = 20

# xlabel = ['cones_fixed','rods_fixed']
axs[0].plot(tr,fev_cnn,'-ko',label='cnn')
axs[0].plot(tr,fev_cones_trainable,'-go',label='cones_train')
# axs[0].plot(tr,fev_rods_fixed,'--gx',label='rods_fixed')
# axs[0].plot(tr,fev_pr_ln,'-o',label='linear PR')
axs[0].set_xlabel('Training data (minutes)')
axs[0].set_ylabel('Median FEV (%) | N = 57 RGCs')
axs[0].legend(loc='best')
axs[0].set_title('PHOTOPIC 80 - 60 ms')

# %% Layer Norm output

idx_mdl_start = 1
idx_mdl_end = 2

x = Input(shape=data_val.X.shape[1:])
n_cells = data_val.y.shape[1]
y = x 

for layer in mdl.layers[idx_mdl_start:idx_mdl_end]:
    y = layer(y)
# y = y[:,-60:]
mdl_ln = Model(x, y, name='LayerNorm')

batch_size = 32
gen = chunker(data_val_select,batch_size=batch_size)    
steps_per_epoch = int(np.ceil(len(data_val_select.X)/batch_size))
norm_inp = mdl_ln.predict_generator(gen,steps=steps_per_epoch)

bins = np.arange(0,2,0.1)
plt.hist(data_val.X[:,-1,:,:].flatten(),bins);plt.show()
plt.hist(norm_inp[:,-1,:,:].flatten(),bins);plt.show()


# %% PR output
idx_mdl_start = 1
idx_mdl_end = 4

x = Input(shape=data_val.X.shape[1:])
n_cells = data_val.y.shape[1]
y = x 

for layer in mdl.layers[idx_mdl_start:idx_mdl_end]:
    y = layer(y)
y = y[:,-60:]
mdl_pr = Model(x, y, name='PRFR')

batch_size = 32
gen = chunker(data_val_select,batch_size=batch_size)    
steps_per_epoch = int(np.ceil(len(data_val_select.X)/batch_size))
pred_rate = mdl_pr.predict_generator(gen,steps=steps_per_epoch)
fac = data_val.X.max()/pred_rate.max()
pred_rate_norm = (pred_rate-pred_rate.min())/(pred_rate.max()-pred_rate.min())

plt.plot(data_val.X[:,-1,20,10]);plt.plot(pred_rate_norm[:,-1,20,10]*2)



stim_flatten = data_val.X[:,-1].reshape(data_val.X.shape[0],-1)
pr_flatten = pred_rate_norm[:,-1].reshape(pred_rate_norm.shape[0],-1)


lags = np.empty(pr_flatten.shape[1])

for i in range(pr_flatten.shape[1]):
    a = stim_flatten[:,i]
    b = pr_flatten[:,i]
    
    corr,lags[i] = cross_corr(a, b)

lags_median = np.median(lags)
print(lags_median)

lags_idx = np.argsort(-lags)
a = lags[lags_idx]

idx = lags_idx[1400]; plt.plot(stim_flatten[:,idx]);plt.plot(pr_flatten[:,idx]*2)


# %% D2: Test model
transferModel = True
if transferModel == True:
    select_mdl_d2 = 'PRFR_CNN2D_NOLN'
    subFold_d2 = subFold
    idx_mdl_start = 1
    idx_mdl_end = 4
    
else:
    select_mdl_d2 = select_mdl
    subFold_d2 = subFold
    idx_currMdl_start = 0#4


val_dataset_2_all = ('CB_CORR_mesopic-Rstar_f4_8ms',)
suffix = '' #'-Rstar_mdl-rieke_s-7.07_p-7.07_e-2.53_k-0.01_h-3_b-25_hc-4_gd-15.5_g-50.0_preproc-rods_norm-0_tb-8_Euler_RF-2'
med_fevs = 0
idx_cells_valid = np.arange(fev_d1_allUnits.shape[0])
fev_stack = fev_d1_allUnits[idx_cells_valid]
                       
v_idx = 0
for v_idx in range(len(val_dataset_2_all)):
    
    val_dataset_2 = val_dataset_2_all[v_idx]+suffix #'scot-30-Rstar'#-Rstar_mdl-rieke_s-25.8898_p-21.1589_e-76.97_k-0.01_h-3_b-47.38_hc-4_gd-28_g-1048.0_preproc-rods_norm-0_tb-8_Euler_RF-2'
    correctMedian = False
    samps_shift_2 = 0#samps_shift
    
    path_dataset_d2 = os.path.join('/home/saad/data/analyses/data_kiersten/',expDates[idx_exp],subFold_d2,'datasets')
    fname_data_train_val_test = os.path.join(path_dataset_d2,(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
    _,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
    resp_orig = resp_orig['train']
    
    
    obs_rate_allStimTrials_d2 = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]
    
    filt_width = filt_temporal_width
    
    if select_mdl_d2[:6]=='CNN_2D' or select_mdl_d2 == 'replaceDense_2D':
        data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
        # data_test = prepare_data_cnn2d(data_test,select_T,np.arange(data_test.y.shape[1]))
    elif select_mdl_d2[:6]=='CNN_3D':
        data_val = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))
        # data_test = prepare_data_cnn3d(data_test,select_T,np.arange(data_test.y.shape[1]))
        
    elif select_mdl_d2[:8]=='PR_CNN2D' or select_mdl_d2[:10]=='PRFR_CNN2D' or select_mdl_d2[:8]=='BP_CNN2D' or select_mdl=='BPFELIX_CNN2D':
        pr_temporal_width = perf_allExps[select_exp][select_mdl][paramFileName]['model_params']['pr_temporal_width']
        data_val = prepare_data_pr_cnn2d(data_val,pr_temporal_width,np.arange(data_val.y.shape[1]))
        obs_rate_allStimTrials_d2 = dataset_rr['stim_0']['val'][:,pr_temporal_width:,:]
    
    elif select_mdl_d2[:8]=='PR_CNN3D':
        pr_temporal_width = perf_allExps[select_exp][select_mdl][paramFileName]['model_params']['pr_temporal_width']
        data_val = prepare_data_cnn3d(data_val,pr_temporal_width,np.arange(data_val.y.shape[1]))
    
    
    if transferModel == True:
        if idx_mdl_end == -1:
            idx_mdl_end = len(mdl.layers)
        x = Input(shape=data_val.X.shape[1:])
        n_cells = data_val.y.shape[1]
        # y = Reshape((x.shape[1],x.shape[-2]*x.shape[-1]),name='start')(x)
        y = x 
    
        for layer in mdl.layers[idx_mdl_start:idx_mdl_end]:
            y = layer(y)
            
        y = y[:,x.shape[1]-filt_temporal_width:,:,:]
        # y = model.models.Normalize_PRDA_GF(units=1)(y)
        for layer in mdl.layers[5:6]:
            y = layer(y)
    
        mdl_d2 = Model(x, y, name='CNN_2D')
        
    elif idx_currMdl_start > 0:     # If we want to remove the beginning layers from the current model
    
        x = Input(shape=data_val.X.shape[1:])
        n_cells = data_val.y.shape[1]
        y = x 
    
        for layer in mdl.layers[idx_currMdl_start:2]:
            if layer.name=='tf.__operators__.getitem':
                y = y[:,x.shape[1]-filt_temporal_width:,:,:]
            # elif layer.name=='tf.__operators__.getitem_1':
            #     y = y[:,x.shape[1]-filt_temporal_width:,:,:,:]
            else:
                y = layer(y)
            
        mdl_d2 = Model(x, y, name=select_mdl_d2)
    else:
        mdl_d2 = mdl
        
    pred_rate = mdl_d2.predict(data_val.X,batch_size=64)

    obs_rate = data_val.y

    
    if correctMedian==True:
        fname_data_train_val_test_d1 = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_1+'.h5'))
        _,_,_,data_quality,_,_,resp_med_d1 = load_h5Dataset(fname_data_train_val_test_d1)
        resp_med_d1 = np.nanmedian(resp_med_d1['train'],axis=0)
        resp_med_d2 = np.nanmedian(resp_orig,axis=0)
        resp_mulFac = resp_med_d2/resp_med_d1;
        
        obs_rate_allStimTrials_d2 = obs_rate_allStimTrials_d2/resp_mulFac
        obs_rate = obs_rate/resp_mulFac
        
        # pred_rate = mdl_d2.predict(data_val.X)
        # pred_rate = pred_rate * resp_mulFac[None,:]
    
        
    # else:       
    
    # pred_rate = mdl_d2.predict(data_val.X)  # /1.4 for 0.3Rstar
    
    
    num_iters = 50
    fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
    predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    
    for i in range(num_iters):
        fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_d2,pred_rate,0,RR_ONLY=False,lag = samps_shift_2)
    
    
    fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
    # fev_d2_allUnits[fev_d2_allUnits<0] = 0
    fracExplainableVar = np.mean(fracExplainableVar,axis=1)
    predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
    rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)
    
    idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
    idx_d2_valid = idx_allUnits
    idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
    idx_d2_valid = idx_allUnits[idx_d2_valid]
    
    # idx_d2_valid = idx_d1_valid
    
    # fev_d2_validUnits = fev_d2_allUnits[idx_d2_valid]
    # predCorr_d2_validUnits = predCorr_d2_allUnits[idx_d2_valid]
    # rrCorr_d2_validUnits = rrCorr_d2_allUnits[idx_d2_valid]
    # # fev_d2_allUnits = fev_d2_allUnits[idx_valid]
    
    fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
    print('FEV = %0.2f' %(fev_d2_medianUnits*100))
    fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
    fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)
    
    predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
    print('R = %0.2f' %(predCorr_d2_medianUnits*100))
    predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
    predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)
    
    rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
    rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
    rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)
    
    # idx_units_sorted = np.argsort(predCorr_d2_allUnits[idx_d2_valid])
    idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
    idx_units_sorted = idx_d2_valid[idx_units_sorted]
    # idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[1],idx_units_sorted[0]]
    # idx_unitsToPred = [26,3,16,32]
    
    # %
    t_start = 10 + 45
    t_dur = obs_rate.shape[0]
    t_end = t_dur-20
    win_display = (t_start,t_start+t_dur)
    font_size_ticks = 14
    
    t_frame = 8
    t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)
    
    col_mdl = ('r')
    lineWidth_mdl = [2]
    lim_y = (0,6)
    fig,axs = plt.subplots(2,2,figsize=(25,10))
    axs = np.ravel(axs)
    fig.suptitle('Training: '+lightLevel_1+' | Prediction: '+ val_dataset_2,fontsize=16)
    
    
    for i in range(len(idx_unitsToPred)):
        l_base, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate[t_start:t_end-samps_shift_2,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
        l_base.set_label('Actual')
        
        l, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate[t_start+samps_shift_2:t_end,idx_unitsToPred[i]],cols_lightLevels[val_dataset_2[:8]],linewidth=lineWidth_mdl[0])
        l.set_label('Predicted: FEV = %.02f, Corr = %0.2f' %(fev_d2_allUnits[idx_unitsToPred[i]],predCorr_d2_allUnits[idx_unitsToPred[i]]))
        
        # axs[i].set_ylim(lim_y)
        axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
        axs[i].set_ylabel('Normalized spike rate (spikes/second)',fontsize=font_size_ticks)
        axs[i].set_title('unit_id: %d' %idx_unitsToPred[i])
        axs[i].legend()
        plt.setp(axs[i].get_xticklabels(), fontsize=font_size_ticks)



    # idx_cells_notvalid = np.union1d(np.setdiff1d(idx_allUnits,idx_d1_valid),np.setdiff1d(idx_allUnits,idx_d2_valid))
    # idx_cells_valid = np.setdiff1d(idx_allUnits,idx_cells_notvalid)
    fev_stack = np.vstack((fev_stack,fev_d2_allUnits[idx_cells_valid]))
    med_fevs = np.vstack((med_fevs,fev_d2_medianUnits))

# % Violin plots
# med_fevs = med_fevs[1:]*100
fev_stack_plot = fev_stack[1:]*100
idx_cells_valid = np.logical_and(fev_stack_plot>-110,fev_stack_plot<110)
fev_stack_plot[~idx_cells_valid] = np.nan
fev_stack_plot = fev_stack_plot.T

med_fevs = np.nanmedian(fev_stack_plot,axis=0)


# %Bar plots of photopic and scotopic light levels
font_size_ticks = 50
font_size_labels = 50


mdls_order = val_dataset_2_all #[val_dataset_1[:8],val_dataset_2[:8]]
# fev_stack = np.vstack((fev_d1_allUnits,fev_d2_allUnits))*100



figure,axs = plt.subplots(1,1,figsize=(20,20))
axs = np.ravel(axs)
figure.suptitle('')

# col_scheme = ('blue','red')
col_scheme = ('b','r','g')

# ax.yaxis.grid(True)
seaborn.violinplot(data=fev_stack_plot,ax=axs[0],palette=col_scheme)

axs[0].set_ylabel('FEV (%)',fontsize=font_size_ticks)
axs[0].set_title('Trained at '+val_dataset_1,fontsize=font_size_labels)
axs[0].set_yticks(np.arange(0,150,10))
axs[0].set_ylim((0,150))
axs[0].set_xticklabels(mdls_order)
for i in range(len(mdls_order)):
    axs[0].text(i+.2,120,'med\n%0.2f%%' %med_fevs[i],fontsize=font_size_labels,color=col_scheme[i])
axs[0].text(-0.4,100,'N = %d RGCs' %fev_stack_plot.shape[0],fontsize=font_size_ticks)
axs[0].tick_params(axis='both',labelsize=font_size_labels)






# %% Sustained, Transient split
unames_valid = perf_allExps[select_exp][select_mdl][paramFileName]['data_quality']['uname_selectedUnits'][idx_d2_valid]
idx_sust = np.arange(0,29)
idx_trans = np.arange(29,64)

idx_sust_valid = np.intersect1d(idx_d2_valid,idx_sust)
idx_trans_valid = np.intersect1d(idx_d2_valid,idx_trans)

fev_sust_allUnits = fev_d2_allUnits[idx_sust_valid]
fev_sust_medianUnits = np.median(fev_sust_allUnits)
fev_sust_stdUnits = np.std(fev_sust_allUnits)
fev_sust_ci = 1.96*(fev_sust_stdUnits/idx_sust_valid.shape[0]**.5)

fev_trans_allUnits = fev_d2_allUnits[idx_trans_valid]
fev_trans_medianUnits = np.median(fev_trans_allUnits)
fev_trans_stdUnits = np.std(fev_trans_allUnits)
fev_trans_ci = 1.96*(fev_trans_stdUnits/idx_trans_valid.shape[0]**.5)

fevs_d1 = [fev_d1_medianUnits,0]
fevs_d2 = [fev_sust_medianUnits,0]
fevs_d3 = [fev_trans_medianUnits,0]

cis_d1 = [fev_d1_ci,0]
cis_d2 = [fev_sust_ci,0]
cis_d3 = [fev_trans_ci,0]

fig,axs = plt.subplots(2,1,figsize=(5,10))
fig.suptitle(select_exp+'\nTraining: '+lightLevel_1,fontsize=16)
font_size_ticks = 16
font_size_labels = 16

col_scheme = (cols_lightLevels[val_dataset_1[:8]],cols_lightLevels[val_dataset_2[:8]])
# ax.yaxis.grid(True)
xpoints = np.atleast_1d((0,1))
xlabel_fev = [select_mdl,'']
axs[0].bar(xpoints-.2,fevs_d1,yerr=cis_d1,align='center',capsize=6,alpha=.7,color=col_scheme[0],width=0.4,label=val_dataset_1[:8])
axs[0].bar(xpoints+.2,fevs_d2,yerr=cis_d2,align='center',capsize=6,alpha=.7,color=col_scheme[1],width=0.4,label=val_dataset_2[:8]+'_sust')
axs[0].bar(xpoints+.6,fevs_d2,yerr=cis_d3,align='center',capsize=6,alpha=.7,color='m',width=0.4,label=val_dataset_2[:8]+'_trans')
axs[0].plot((-0.5,len(xlabel_fev)-0.5),(np.nanmax(fevs_d1),np.nanmax(fevs_d1)),color='green')
axs[0].set_xticks(xpoints)#(2*np.arange(0,fev_d1_medianUnits_allMdls.shape[0]))
axs[0].set_xticklabels(xlabel_fev)

axs[0].set_ylabel('FEV',fontsize=font_size_ticks)
axs[0].set_title('',fontsize=font_size_ticks)
axs[0].set_yticks(np.arange(-1,1.1,.1))
axs[0].set_ylim((0,1.1))
# axs[0].text(.75,.45,'N = %d RGCs' %fev_allUnits.shape[1],fontsize=font_size_ticks)
axs[0].tick_params(axis='both',labelsize=16)
axs[0].legend(loc='best',fontsize=font_size_labels)



# %% D1: Heat maps
RECALC_PERF = 1
select_exp = '20230725C'
select_mdl = models_all[0]
pr_params_name = ''
select_param_x = 'TRSAMPS' #'C2_n'
select_param_y = 'C3_n' #'C3_n'
select_param_z = 'C1_n'
thresh_fev = -1000

params_mdl = params_allExps[select_exp][select_mdl]

select_U = 57
select_P = 0
select_T = np.array([80]) #np.unique(params_mdl['T'])
select_BN = 1
select_MP = 2
# select_TR = 1
select_C1_n = np.unique(params_mdl['C1_n'])#13
select_C1_s = np.unique(params_mdl['C1_s'])
select_C1_3d = np.unique(params_mdl['C1_3d'])
select_C2_n = np.unique(params_mdl['C2_n'])#13
select_C2_s = np.unique(params_mdl['C2_s'])
select_C2_3d = np.unique(params_mdl['C2_3d'])
select_C3_n = np.unique(params_mdl['C3_n'])#25
select_C3_s = np.unique(params_mdl['C3_s'])
select_C3_3d = np.unique(params_mdl['C3_3d'])
select_TRSAMPS = np.unique(params_mdl['TRSAMPS'])

idx_interest = np.in1d(params_mdl['U'],select_U)
idx_interest = idx_interest & np.in1d(params_mdl['T'],select_T)
idx_interest = idx_interest & np.in1d(params_mdl['BN'],select_BN)
idx_interest = idx_interest & np.in1d(params_mdl['MP'],select_MP)
# idx_interest = idx_interest & np.in1d(params_mdl['TR'],select_TR)

idx_interest = idx_interest & np.in1d(params_mdl['C1_n'],select_C1_n)
idx_interest = idx_interest & np.in1d(params_mdl['C1_s'],select_C1_s)
idx_interest = idx_interest & np.in1d(params_mdl['C1_3d'],select_C1_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C2_n'],select_C2_n)
idx_interest = idx_interest & np.in1d(params_mdl['C2_s'],select_C2_s)
idx_interest = idx_interest & np.in1d(params_mdl['C2_3d'],select_C2_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C3_n'],select_C3_n)
idx_interest = idx_interest & np.in1d(params_mdl['C3_s'],select_C3_s)
idx_interest = idx_interest & np.in1d(params_mdl['C3_3d'],select_C3_3d)

idx_interest = idx_interest  &  np.in1d(params_mdl['TRSAMPS'],select_TRSAMPS)

# make array of all fevs for min max vals
fev_grand = ([])
for i in perf_allExps[select_exp][select_mdl]:
    if perf_allExps[select_exp][select_mdl][i]['model_performance']['num_trials'] == 1:
        rgb = perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_bestEpoch_allTr']
    else:
        rgb = np.mean(perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
    
    fev_grand.append(rgb)
fev_min, fev_max = np.round(np.nanmin(fev_grand),2), np.round(np.nanmax(fev_grand),2)
fev_min, fev_max = fev_min-.01, fev_max+.01


# plot heatmap

fev_heatmap = np.zeros((len(eval('select_'+select_param_y)),len(eval('select_'+select_param_x)),len(eval('select_'+select_param_z))))
fev_heatmap[:] = np.nan

totalCombs = sum(idx_interest)
idx_interest = np.where(idx_interest)[0]

i=idx_interest[0]
for i in idx_interest:
    idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
    idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
    idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    
    if RECALC_PERF == 0:
        if perf_allExps[select_exp][select_mdl][paramName]['model_performance']['num_trials'] == 1:
            rgb = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr']
        else:
            rgb = np.mean(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
        rgb = np.nanmedian(rgb,axis=0)
        if rgb>thresh_fev:
            fev_heatmap[idx_y,idx_x,idx_z] = rgb

    else:
               
        idx_bestTrial = np.argsort(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
        select_TR = idx_bestTrial[-1] + 1
        mdlFolder = paramName+'_TR-%02d' % select_TR
        idx_bestEpoch = len(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])-1
        # idx_bestEpoch = np.argmax(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])
        # idx_bestEpoch = np.argsort(np.squeeze(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial]))[-2]
        fname_bestWeight = 'weights_'+mdlFolder+'_epoch-%03d' % (idx_bestEpoch+1)

        
        curr_params = getModelParams(mdlFolder)
        # get datasets if recalc_perf==1
        if RECALC_PERF==1:
            val_dataset_1 = lightLevel_1[0]
            
            fname_data_train_val_test = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_1+'.h5'))
            rgb = load_h5Dataset(fname_data_train_val_test,nsamps_val=0.5,nsamps_test=0.1,nsamps_train=0.05,LOAD_TR=True)
            data_val = rgb[1];
            data_quality = rgb[3]

            if select_mdl[:6]=='CNN_2D' or select_mdl == 'replaceDense_2D' or select_mdl[:5]=='CNN2D':
                data_val = prepare_data_cnn2d(data_val,curr_params['T'],np.arange(data_val.y.shape[1]))

            elif 'PR' in select_mdl:
                data_val = prepare_data_cnn2d(data_val,curr_params['P'],np.arange(data_val.y.shape[1]))

            samps_shift = 0
            
            if data_val.y.ndim==3:
                obs_rate_allStimTrials_d1 = obs_rate_allStimTrials_d1
            else:
                obs_rate_allStimTrials_d1 = data_val.y

            # obs_rate = obs_rate_allStimTrials_d1.copy()
            obs_rate = data_val.y
            if 'var_noise' in data_quality:
                obs_noise = data_quality['var_noise'] #estimate_noise(data_test.y_trials)
            else:
                obs_noise = 0




        if 'PR' in select_mdl:
            path_model = os.path.join(path_mdl_drive,select_mdl,pr_params_name,mdlFolder)
        else:
            path_model = os.path.join(path_mdl_drive,select_mdl,mdlFolder)
            try:
                tf.compat.v1.disable_eager_execution()
            except:
                pass

        mdl = load(os.path.join(path_model,mdlFolder))

        try:
            mdl.load_weights(os.path.join(path_model,fname_bestWeight))
        except:
            mdl.load_weights(os.path.join(path_model,fname_bestWeight+'.h5'))
        weights_dict = get_weightsDict(mdl)

        # pred_rate = mdl.predict(data_val.X)
        batch_size = 16
        gen = chunker(data_val,batch_size=batch_size)    
        steps_per_epoch = int(np.ceil(len(data_val.X)/batch_size))
        pred_rate = mdl.predict_generator(gen,steps=steps_per_epoch)
        # pred_rate = mdl.predict(data_val.X,batch_size = 4)


        num_iters = 1
        fev_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
        fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
        predCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
        rrCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
        
        for j in range(num_iters):
            fev_d1_allUnits[:,j], fracExplainableVar[:,j], predCorr_d1_allUnits[:,j], rrCorr_d1_allUnits[:,j] = model_evaluate_new(
                                                                            obs_rate_allStimTrials_d1.copy(),pred_rate.copy(),
                                                                            0,RR_ONLY=False,lag = samps_shift,obs_noise=obs_noise)        
        
        fev_d1_allUnits = np.mean(fev_d1_allUnits,axis=1)
    
        idx_allUnits = np.arange(fev_d1_allUnits.shape[0])
        idx_d1_valid = idx_allUnits
        # idx_d1_valid = np.logical_and(fev_d1_allUnits>-1,fev_d1_allUnits<1.1)
        # idx_d1_valid = idx_allUnits[idx_d1_valid]
        
        fev_d1_medianUnits = np.median(fev_d1_allUnits[idx_d1_valid])
        if fev_d1_medianUnits>thresh_fev:
            fev_heatmap[idx_y,idx_x,idx_z] = fev_d1_medianUnits

    

font_size_ticks = 12
fig_rows = 1
fig_cols = int(np.ceil(len(eval('select_'+select_param_z))/fig_rows))
fig,axs = plt.subplots(fig_rows,fig_cols,figsize=(10,5))
axs = np.ravel(axs)
axs = axs[:len(eval('select_'+select_param_z))]

color_map = plt.cm.get_cmap('autumn')
reversed_color_map = color_map.reversed()

for l in range(len(eval('select_'+select_param_z))):
    param_z = eval('select_'+select_param_z)[l]

    # im = axs[l].imshow(fev_heatmap_sorted[:,:,l],cmap=reversed_color_map,vmin = fev_min, vmax = fev_max)
    im = axs[l].imshow(fev_heatmap[:,:,l],cmap=reversed_color_map,vmin = fev_min, vmax = fev_max)
    axs[l].set_yticks(range(0,len(eval('select_'+select_param_y))))
    axs[l].set_yticklabels(eval('select_'+select_param_y),fontsize=font_size_ticks)
    axs[l].set_xticks(range(0,len(eval('select_'+select_param_x))))
    axs[l].set_xticklabels(eval('select_'+select_param_x),fontsize=font_size_ticks)
    axs[l].set_xlabel(select_param_x)
    axs[l].set_ylabel(select_param_y)
    axs[l].set_title(select_param_z+' = '+str(param_z))
    # axs[l].set_aspect('equal', adjustable='datalim')
fig.colorbar(im)
plt.setp(axs,aspect='equal')

rgb = np.nanmax(fev_heatmap)
[max_y,max_x,max_z] = np.where(fev_heatmap==rgb)

max_param_x = eval('select_'+select_param_x)[max_x][0]
max_param_y = eval('select_'+select_param_y)[max_y][0]
max_param_z = eval('select_'+select_param_z)[max_z][0]
print('Max params:\n%s=%d\n%s=%d\n%s=%d\n' %(select_param_x,max_param_x,select_param_y,max_param_y,select_param_z,max_param_z))
print('Max FEV: %0.2f' %(fev_heatmap[max_y,max_x,max_z]*100))

# top 10 params
fev_heatmap_sorted_idx = np.argsort(fev_heatmap,axis=None)
fev_heatmap_sorted_vals = np.sort(fev_heatmap,axis=None)
idx_nans = np.isnan(fev_heatmap_sorted_vals)
fev_heatmap_sorted_idx = fev_heatmap_sorted_idx[~idx_nans]
fev_heatmap_sorted_vals = fev_heatmap_sorted_vals[~idx_nans]
fev_heatmap_sorted_idx = np.flip(fev_heatmap_sorted_idx)
fev_heatmap_sorted_vals = np.flip(fev_heatmap_sorted_vals)

N = 10
params_top = np.zeros((N,4))
for i in range(N):
    curr_fev = fev_heatmap_sorted_vals[i]
    [curr_y,curr_x,curr_z] = np.where(fev_heatmap==curr_fev)
    curr_param_x = eval('select_'+select_param_x)[curr_x][0]
    curr_param_y = eval('select_'+select_param_y)[curr_y][0]
    curr_param_z = eval('select_'+select_param_z)[curr_z][0]
    
    params_top[i,:] = [curr_fev,curr_param_x,curr_param_y,curr_param_z]
    print('FEV: %0.2f - Params:%s=%02d, %s=%02d, %s=%02d' %(curr_fev*100,select_param_x,curr_param_x,select_param_y,curr_param_y,select_param_z,curr_param_z))



N = 20
paramNames_all = []
fev_all = []
i=0
for i in range(len(perf_paramNames)):
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    paramNames_all.append(paramName)
    curr_fev = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_bestEpoch_allTr']
    fev_all.append(curr_fev)
fev_all = np.squeeze(np.asarray(fev_all))

idx_sorted = np.argsort(fev_all)
models_top = []
for i in idx_sorted[-N:]:
    models_top.append(paramNames_all[i])

fevs_sorted = fev_all[idx_sorted[-N:]]

for i in range(N):
    print('FEV: %0.2f | model: %s'%(fevs_sorted[i],models_top[i]))

# %% Evaluate grid search
'''
Evaluate without calculating performance file
'''
RECALC_PERF = 0
select_exp = '20230725C'
select_mdl = models_all[0]
pr_params_name = ''
select_param_x = 'C2_n' #'C2_n'
select_param_y = 'C3_n' #'C3_n'
select_param_z = 'C1_n'
thresh_fev = -1000

params_mdl = params_allExps[select_exp][select_mdl]

select_U = 57
select_P = 0
select_T = np.array([60]) #np.unique(params_mdl['T'])
select_BN = 1
select_MP = 2
# select_TR = 1
select_C1_n = np.unique(params_mdl['C1_n'])#13
select_C1_s = np.unique(params_mdl['C1_s'])
select_C1_3d = np.unique(params_mdl['C1_3d'])
select_C2_n = np.unique(params_mdl['C2_n'])#13
select_C2_s = np.unique(params_mdl['C2_s'])
select_C2_3d = np.unique(params_mdl['C2_3d'])
select_C3_n = np.unique(params_mdl['C3_n'])#25
select_C3_s = np.unique(params_mdl['C3_s'])
select_C3_3d = np.unique(params_mdl['C3_3d'])
select_TRSAMPS = np.unique(params_mdl['TRSAMPS'])

idx_interest = np.in1d(params_mdl['U'],select_U)
idx_interest = idx_interest & np.in1d(params_mdl['T'],select_T)
idx_interest = idx_interest & np.in1d(params_mdl['BN'],select_BN)
idx_interest = idx_interest & np.in1d(params_mdl['MP'],select_MP)
# idx_interest = idx_interest & np.in1d(params_mdl['TR'],select_TR)

idx_interest = idx_interest & np.in1d(params_mdl['C1_n'],select_C1_n)
idx_interest = idx_interest & np.in1d(params_mdl['C1_s'],select_C1_s)
idx_interest = idx_interest & np.in1d(params_mdl['C1_3d'],select_C1_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C2_n'],select_C2_n)
idx_interest = idx_interest & np.in1d(params_mdl['C2_s'],select_C2_s)
idx_interest = idx_interest & np.in1d(params_mdl['C2_3d'],select_C2_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C3_n'],select_C3_n)
idx_interest = idx_interest & np.in1d(params_mdl['C3_s'],select_C3_s)
idx_interest = idx_interest & np.in1d(params_mdl['C3_3d'],select_C3_3d)

idx_interest = idx_interest  &  np.in1d(params_mdl['TRSAMPS'],select_TRSAMPS)

# make array of all fevs for min max vals
fev_grand = ([])
for i in perf_allExps[select_exp][select_mdl]:
    if perf_allExps[select_exp][select_mdl][i]['model_performance']['num_trials'] == 1:
        rgb = perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_bestEpoch_allTr']
    else:
        rgb = np.mean(perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
    
    fev_grand.append(rgb)
fev_min, fev_max = np.round(np.nanmin(fev_grand),2), np.round(np.nanmax(fev_grand),2)
fev_min, fev_max = fev_min-.01, fev_max+.01


# plot heatmap

fev_heatmap = np.zeros((len(eval('select_'+select_param_y)),len(eval('select_'+select_param_x)),len(eval('select_'+select_param_z))))
fev_heatmap[:] = np.nan

totalCombs = sum(idx_interest)
idx_interest = np.where(idx_interest)[0]

i=idx_interest[0]
for i in idx_interest:
    idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
    idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
    idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    
    if RECALC_PERF == 0:
        if perf_allExps[select_exp][select_mdl][paramName]['model_performance']['num_trials'] == 1:
            rgb = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr']
        else:
            rgb = np.mean(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
        rgb = np.nanmedian(rgb,axis=0)
        if rgb>thresh_fev:
            fev_heatmap[idx_y,idx_x,idx_z] = rgb

    else:
               
        idx_bestTrial = np.argsort(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
        select_TR = idx_bestTrial[-1] + 1
        mdlFolder = paramName+'_TR-%02d' % select_TR
        idx_bestEpoch = len(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])-1
        # idx_bestEpoch = np.argmax(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])
        # idx_bestEpoch = np.argsort(np.squeeze(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial]))[-2]
        fname_bestWeight = 'weights_'+mdlFolder+'_epoch-%03d' % (idx_bestEpoch+1)

        
        curr_params = getModelParams(mdlFolder)
        # get datasets if recalc_perf==1
        if RECALC_PERF==1:
            val_dataset_1 = lightLevel_1[0]
            
            fname_data_train_val_test = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_1+'.h5'))
            rgb = load_h5Dataset(fname_data_train_val_test,nsamps_val=0.5,nsamps_test=0.1,nsamps_train=0.05,LOAD_TR=True)
            data_val = rgb[1];
            data_quality = rgb[3]

            if select_mdl[:6]=='CNN_2D' or select_mdl == 'replaceDense_2D' or select_mdl[:5]=='CNN2D':
                data_val = prepare_data_cnn2d(data_val,curr_params['T'],np.arange(data_val.y.shape[1]))

            elif 'PR' in select_mdl:
                data_val = prepare_data_cnn2d(data_val,curr_params['P'],np.arange(data_val.y.shape[1]))

            samps_shift = 0
            
            if data_val.y.ndim==3:
                obs_rate_allStimTrials_d1 = obs_rate_allStimTrials_d1
            else:
                obs_rate_allStimTrials_d1 = data_val.y

            # obs_rate = obs_rate_allStimTrials_d1.copy()
            obs_rate = data_val.y
            if 'var_noise' in data_quality:
                obs_noise = data_quality['var_noise'] #estimate_noise(data_test.y_trials)
            else:
                obs_noise = 0




        if 'PR' in select_mdl:
            path_model = os.path.join(path_mdl_drive,select_mdl,pr_params_name,mdlFolder)
        else:
            path_model = os.path.join(path_mdl_drive,select_mdl,mdlFolder)
            try:
                tf.compat.v1.disable_eager_execution()
            except:
                pass

        mdl = load(os.path.join(path_model,mdlFolder))

        try:
            mdl.load_weights(os.path.join(path_model,fname_bestWeight))
        except:
            mdl.load_weights(os.path.join(path_model,fname_bestWeight+'.h5'))
        weights_dict = get_weightsDict(mdl)

        # pred_rate = mdl.predict(data_val.X)
        batch_size = 16
        gen = chunker(data_val,batch_size=batch_size)    
        steps_per_epoch = int(np.ceil(len(data_val.X)/batch_size))
        pred_rate = mdl.predict_generator(gen,steps=steps_per_epoch)
        # pred_rate = mdl.predict(data_val.X,batch_size = 4)


        num_iters = 1
        fev_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
        fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
        predCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
        rrCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
        
        for j in range(num_iters):
            fev_d1_allUnits[:,j], fracExplainableVar[:,j], predCorr_d1_allUnits[:,j], rrCorr_d1_allUnits[:,j] = model_evaluate_new(
                                                                            obs_rate_allStimTrials_d1.copy(),pred_rate.copy(),
                                                                            0,RR_ONLY=False,lag = samps_shift,obs_noise=obs_noise)        
        
        fev_d1_allUnits = np.mean(fev_d1_allUnits,axis=1)
    
        idx_allUnits = np.arange(fev_d1_allUnits.shape[0])
        idx_d1_valid = idx_allUnits
        # idx_d1_valid = np.logical_and(fev_d1_allUnits>-1,fev_d1_allUnits<1.1)
        # idx_d1_valid = idx_allUnits[idx_d1_valid]
        
        fev_d1_medianUnits = np.median(fev_d1_allUnits[idx_d1_valid])
        if fev_d1_medianUnits>thresh_fev:
            fev_heatmap[idx_y,idx_x,idx_z] = fev_d1_medianUnits

    

font_size_ticks = 12
fig_rows = 1
fig_cols = int(np.ceil(len(eval('select_'+select_param_z))/fig_rows))
fig,axs = plt.subplots(fig_rows,fig_cols,figsize=(10,5))
axs = np.ravel(axs)
axs = axs[:len(eval('select_'+select_param_z))]

color_map = plt.cm.get_cmap('autumn')
reversed_color_map = color_map.reversed()

for l in range(len(eval('select_'+select_param_z))):
    param_z = eval('select_'+select_param_z)[l]

    # im = axs[l].imshow(fev_heatmap_sorted[:,:,l],cmap=reversed_color_map,vmin = fev_min, vmax = fev_max)
    im = axs[l].imshow(fev_heatmap[:,:,l],cmap=reversed_color_map,vmin = fev_min, vmax = fev_max)
    axs[l].set_yticks(range(0,len(eval('select_'+select_param_y))))
    axs[l].set_yticklabels(eval('select_'+select_param_y),fontsize=font_size_ticks)
    axs[l].set_xticks(range(0,len(eval('select_'+select_param_x))))
    axs[l].set_xticklabels(eval('select_'+select_param_x),fontsize=font_size_ticks)
    axs[l].set_xlabel(select_param_x)
    axs[l].set_ylabel(select_param_y)
    axs[l].set_title(select_param_z+' = '+str(param_z))
    # axs[l].set_aspect('equal', adjustable='datalim')
fig.colorbar(im)
plt.setp(axs,aspect='equal')

rgb = np.nanmax(fev_heatmap)
[max_y,max_x,max_z] = np.where(fev_heatmap==rgb)

max_param_x = eval('select_'+select_param_x)[max_x][0]
max_param_y = eval('select_'+select_param_y)[max_y][0]
max_param_z = eval('select_'+select_param_z)[max_z][0]
print('Max params:\n%s=%d\n%s=%d\n%s=%d\n' %(select_param_x,max_param_x,select_param_y,max_param_y,select_param_z,max_param_z))
print('Max FEV: %0.2f' %(fev_heatmap[max_y,max_x,max_z]*100))

# top 10 params
fev_heatmap_sorted_idx = np.argsort(fev_heatmap,axis=None)
fev_heatmap_sorted_vals = np.sort(fev_heatmap,axis=None)
idx_nans = np.isnan(fev_heatmap_sorted_vals)
fev_heatmap_sorted_idx = fev_heatmap_sorted_idx[~idx_nans]
fev_heatmap_sorted_vals = fev_heatmap_sorted_vals[~idx_nans]
fev_heatmap_sorted_idx = np.flip(fev_heatmap_sorted_idx)
fev_heatmap_sorted_vals = np.flip(fev_heatmap_sorted_vals)

N = 10
params_top = np.zeros((N,4))
for i in range(N):
    curr_fev = fev_heatmap_sorted_vals[i]
    [curr_y,curr_x,curr_z] = np.where(fev_heatmap==curr_fev)
    curr_param_x = eval('select_'+select_param_x)[curr_x][0]
    curr_param_y = eval('select_'+select_param_y)[curr_y][0]
    curr_param_z = eval('select_'+select_param_z)[curr_z][0]
    
    params_top[i,:] = [curr_fev,curr_param_x,curr_param_y,curr_param_z]
    print('FEV: %0.2f - Params:%s=%02d, %s=%02d, %s=%02d' %(curr_fev*100,select_param_x,curr_param_x,select_param_y,curr_param_y,select_param_z,curr_param_z))



N = 20
paramNames_all = []
fev_all = []
i=0
for i in range(len(perf_paramNames)):
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    paramNames_all.append(paramName)
    curr_fev = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_bestEpoch_allTr']
    fev_all.append(curr_fev)
fev_all = np.squeeze(np.asarray(fev_all))

idx_sorted = np.argsort(fev_all)
models_top = []
for i in idx_sorted[-N:]:
    models_top.append(paramNames_all[i])

fevs_sorted = fev_all[idx_sorted[-N:]]

for i in range(N):
    print('FEV: %0.2f | model: %s'%(fevs_sorted[i],models_top[i]))


# %% D2: heatmaps
val_dataset_2 = val_dataser_1 #photopic_preproc_cones_norm_1'

select_exp = 'retina1'
select_mdl = 'CNN_2D'
select_param_x = 'C1_n'
select_param_y = 'C2_n'
select_param_z = 'C3_n'
thresh_fev = 0.0

params_mdl = params_allExps[select_exp][select_mdl]

select_U = 0
select_T = 60
select_BN = 1
select_MP = 0
# select_TR = 1
select_C1_n = np.unique(params_mdl['C1_n'])  #np.atleast_1d(13)#np.unique(params_mdl['C1_n'])#13
select_C1_s = np.unique(params_mdl['C1_s'])
select_C1_3d = np.unique(params_mdl['C1_3d'])
select_C2_n = np.unique(params_mdl['C2_n'])#13
select_C2_s = np.unique(params_mdl['C2_s'])
select_C2_3d = np.unique(params_mdl['C2_3d'])
select_C3_n = np.unique(params_mdl['C3_n'])#25
select_C3_s = np.unique(params_mdl['C3_s'])
select_C3_3d = np.unique(params_mdl['C3_3d'])

idx_interest = np.in1d(params_mdl['U'],select_U)
idx_interest = idx_interest & np.in1d(params_mdl['T'],select_T)
idx_interest = idx_interest & np.in1d(params_mdl['BN'],select_BN)
idx_interest = idx_interest & np.in1d(params_mdl['MP'],select_MP)
# idx_interest = idx_interest & np.in1d(params_mdl['TR'],select_TR)

idx_interest = idx_interest & np.in1d(params_mdl['C1_n'],select_C1_n)
idx_interest = idx_interest & np.in1d(params_mdl['C1_s'],select_C1_s)
idx_interest = idx_interest & np.in1d(params_mdl['C1_3d'],select_C1_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C2_n'],select_C2_n)
idx_interest = idx_interest & np.in1d(params_mdl['C2_s'],select_C2_s)
idx_interest = idx_interest & np.in1d(params_mdl['C2_3d'],select_C2_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C3_n'],select_C3_n)
idx_interest = idx_interest & np.in1d(params_mdl['C3_s'],select_C3_s)
idx_interest = idx_interest & np.in1d(params_mdl['C3_3d'],select_C3_3d)


fev_d2_heatmap = np.zeros((len(eval('select_'+select_param_y)),len(eval('select_'+select_param_x)),len(eval('select_'+select_param_z))))
fev_d2_heatmap[:] = np.nan

totalCombs = sum(idx_interest)
idx_interest = np.where(idx_interest)[0]
#
for i in idx_interest:
    idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
    idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
    idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    
    
    idx_bestTrial = np.argsort(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
    
    select_TR = idx_bestTrial[-1] + 1
    
    
    mdlFolder = paramName+'_TR-%02d' % select_TR
    
    path_model = os.path.join(path_mdl_drive,select_mdl,mdlFolder)
    
    try:
        mdl = load(os.path.join(path_model,mdlFolder))
        # idx_bestEpoch = np.argmax(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial[-1]])
        # fname_bestEpoch = os.path.join(path_model,'weights_'+mdlFolder+'_epoch-%03d.h5'%idx_bestEpoch)
        # mdl.load_weights(fname_bestEpoch)
       
        fname_data_train_val_test = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
        _,data_val,data_test,data_quality,dataset_rr,_,_ = load_h5Dataset(fname_data_train_val_test)
        if select_mdl[:6]=='CNN_2D':
            data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
        else:
            data_val = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))

        resp_median_scotopic = data_quality['resp_median_allUnits']
        obs_rate_allStimTrials_d2 = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]
      
        samps_shift = 0
        
        obs_rate = data_val.y
        pred_rate = mdl.predict(data_val.X)
      
        
        num_iters = 1
        fev_scot_allUnits = np.empty((pred_rate.shape[1],num_iters))
        
        for i in range(num_iters):
            # fev_scot_allUnits[:,i],_,_,_ = model_evaluate_new(obs_rate_allStimTrials_d2,pred_rate,0,RR_ONLY=False,lag = samps_shift)
            _,_,fev_scot_allUnits[:,i],_ = model_evaluate_new(obs_rate_allStimTrials_d2,pred_rate,0,RR_ONLY=False,lag = samps_shift)       # correlation
        fev_scot_allUnits = np.mean(fev_scot_allUnits,axis=1)
    
        idx_scotopic_valid = np.logical_and(fev_scot_allUnits>0,fev_scot_allUnits<1.1)
        fev_scot_allUnits = fev_scot_allUnits[idx_scotopic_valid]
        
        fev_scot_medianUnits = np.nanmedian(fev_scot_allUnits)
       
        
        if fev_scot_medianUnits>thresh_fev:
            fev_d2_heatmap[idx_y,idx_x,idx_z] = fev_scot_medianUnits
    except:
        fev_d2_heatmap[idx_y,idx_x,idx_z] = np.nan
        
#%
fev_min, fev_max = np.round(np.nanmin(fev_d2_heatmap),2), np.round(np.nanmax(fev_d2_heatmap),2)
# fev_min, fev_max = 0.56,0.7
# %

font_size_ticks = 10
fig_rows = 3
fig_cols = int(np.ceil(len(eval('select_'+select_param_z))/fig_rows))
fig,axs = plt.subplots(fig_rows,fig_cols,figsize=(20,15))
axs = np.ravel(axs)
axs = axs[:len(eval('select_'+select_param_z))]

color_map = plt.cm.get_cmap('hot')
reversed_color_map = color_map.reversed()

for l in range(len(eval('select_'+select_param_z))):
    param_z = eval('select_'+select_param_z)[l]

    im = axs[l].imshow(fev_d2_heatmap[:,:,l],cmap=reversed_color_map,vmin = fev_min, vmax = fev_max)
    axs[l].set_yticks(range(0,len(eval('select_'+select_param_y))))
    axs[l].set_yticklabels(eval('select_'+select_param_y),fontsize=font_size_ticks)
    axs[l].set_xticks(range(0,len(eval('select_'+select_param_x))))
    axs[l].set_xticklabels(eval('select_'+select_param_x),fontsize=font_size_ticks)
    axs[l].set_xlabel(select_param_x)
    axs[l].set_ylabel(select_param_y)
    axs[l].set_title(select_param_z+' = '+str(param_z))
    # axs[l].set_aspect('equal', adjustable='datalim')
fig.colorbar(im)

plt.setp(axs,aspect='equal')



# %% Feature maps
select_exp = 'retina1'
select_mdl = models_all[0] #'CNN_2D' #'CNN_2D_chansVary'#'CNN_2D_filtsVary'
params_mdl = params_allExps[select_exp][select_mdl]

select_LR=0.001
select_U = 0#0.15
select_P = 180#180
select_T = 120#120
select_BN = 1
select_MP = 0
# select_TR = 1
select_C1_n = 1#2#13 #13 #20
select_C1_s = 3
select_C1_3d = 0#50#25
select_C2_n = 40#26 #26 #24
select_C2_s = 3#2#2
select_C2_3d = 0#10#5
select_C3_n = 36 #24 #22
select_C3_s = 2#1#1
select_C3_3d = 0#62#32

paramFileName,_ = modelFileName(U=select_U,P=select_P,T=select_T,BN=select_BN,MP=select_MP,LR=select_LR,
                 C1_n=select_C1_n,C1_s=select_C1_s,C1_3d=select_C1_3d,
                 C2_n=select_C2_n,C2_s=select_C2_s,C2_3d=select_C2_3d,
                 C3_n=select_C3_n,C3_s=select_C3_s,C3_3d=select_C3_3d,with_TR=False)


idx_bestTrial = np.nanargmax(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
idx_bestEpoch = np.nanargmax(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])
trial_num = perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['trial_id'][idx_bestTrial]
plt.plot(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])

select_TR = int(trial_num)

mdlFolder = paramFileName+'_TR-%02d' % select_TR
path_model = os.path.join(path_mdl_drive,select_mdl,mdlFolder)
mdl = load(os.path.join(path_model,mdlFolder))
fname_bestWeight = 'weights_'+mdlFolder+'_epoch-%03d' % (idx_bestEpoch+1)
try:
    mdl.load_weights(os.path.join(path_model,fname_bestWeight))
except:
    mdl.load_weights(os.path.join(path_model,fname_bestWeight+'.h5'))


fname_data_train_val_test = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+lightLevel_1+'.h5'))
data_train,_,_,_,_,_,_ = load_h5Dataset(fname_data_train_val_test)
if select_mdl[:6]=='CNN_2D':
    data_train = prepare_data_cnn2d(data_train,select_T,np.arange(data_train.y.shape[1]))
    chunk_size = data_train.y.shape[0]
    
elif select_mdl[:8]=='PR_CNN2D' or select_mdl[:10]=='PRFR_CNN2D' or select_mdl[:8]=='BP_CNN2D' or select_mdl=='BPFELIX_CNN2D':
    pr_temporal_width = perf_allExps[select_exp][select_mdl][paramFileName]['model_params']['pr_temporal_width']
    data_train = prepare_data_pr_cnn2d(data_train,pr_temporal_width,np.arange(data_train.y.shape[1]))
    chunk_size = 50000 # data_train.y.shape[0]

else:
    data_train = prepare_data_cnn3d(data_train,select_T,np.arange(data_train.y.shape[1]))
    chunk_size = 10000

stim_size = np.array(mdl.input.shape[-2:])
idx_extra = 2
# strongestUnit = np.array([9,15])

mdl_params = {
    'chan_n': select_C1_n,
    'filt_temporal_width': pr_temporal_width,
    'chan_s': 1 #select_C1_s
    }

layer_name = 'permute'

# import multiprocessing as mp
# pool = mp.Pool(mp.cpu_count())

strongestUnit = np.array([8,7,100])

_ = gc.collect()
rwa_mean = featureMaps.get_featureMaps(data_train.X,mdl,mdl_params,layer_name,strongestUnit,chunk_size,PARALLEL_PROC=False)
# pool.close()

# 
font_size_ticks = 10
fig_rows = 4
fig_cols = int(np.ceil(rwa_mean.shape[0]/fig_rows))
t_frame = 8
t_axis = np.flip(-1*np.arange(0,pr_temporal_width*t_frame,t_frame))
if strongestUnit[2]>-1:
    t_axis = t_axis + ((select_T-strongestUnit[2])*t_frame)

fig_spat,axs_spat = plt.subplots(fig_rows,fig_cols,figsize=(15,12))
fig_spat.suptitle('Layer: '+layer_name+' | Spatial features',fontsize=16)
axs_spat = np.ravel(axs_spat)
axs_spat = axs_spat[:rwa_mean.shape[0]]

fig_temp,axs_temp = plt.subplots(fig_rows,fig_cols,figsize=(20,12))
fig_temp.suptitle('Layer: '+layer_name+' | Temporal features',fontsize=16)
axs_temp = np.ravel(axs_temp)
axs_temp = axs_temp[:rwa_mean.shape[0]]

temporal_feature_all = np.zeros((rwa_mean.shape[0],rwa_mean.shape[1]))

for i in range(rwa_mean.shape[0]):
    spatial_feature, temporal_feature = decompose(rwa_mean[i,:,:,:])
    im = axs_spat[i].imshow(spatial_feature,cmap='bwr')
    axs_temp[i].plot(t_axis,temporal_feature)
    axs_temp[i].set_title('Filt # '+str(i+1))
    temporal_feature_all[i,:] = temporal_feature
    


fig_spat.colorbar(im)   # axs_temp[i].yticks([])

# Correlations between filters
filt_simMatrix = np.zeros((temporal_feature_all.shape[0],temporal_feature_all.shape[0]))
filt_simMatrix_lag = np.zeros((temporal_feature_all.shape[0],temporal_feature_all.shape[0]))

for i in range(temporal_feature_all.shape[0]):
    for j in range(temporal_feature_all.shape[0]):
        # [filt_simMatrix[i,j],filt_simMatrix_lag[i,j]] = featureMaps.cross_corr(temporal_feature_all[i],temporal_feature_all[j])
        [filt_simMatrix[i,j],filt_simMatrix_lag[i,j]] = featureMaps.xcorr(temporal_feature_all[i],temporal_feature_all[j])

color_map = plt.cm.get_cmap('autumn')
reversed_color_map = color_map.reversed()

fig_spat,axs_spat = plt.subplots(1,1,figsize=(5,5))
fig_spat.suptitle('Layer: '+layer_name+' | Temporal filt cross-corr',fontsize=16)
axs_spat.imshow(filt_simMatrix,cmap=reversed_color_map,vmin = 0, vmax = 1.1)
axs_spat.set_xticks(np.arange(0+.5,temporal_feature_all.shape[0]+.5))
axs_spat.set_xticklabels(np.arange(1,temporal_feature_all.shape[0]+1),fontsize=font_size_ticks)
axs_spat.set_yticks(np.arange(0+.5,temporal_feature_all.shape[0]+.5))
axs_spat.set_yticklabels(range(1,temporal_feature_all.shape[0]+1),fontsize=font_size_ticks)
axs_spat.grid()
        
# %% Bipolar Filters
from model.models import generate_simple_filter_multichan, conv_oper_multichan, dart_model_diff
tf.compat.v1.disable_eager_execution()


weights_dict = get_weightsDict(mdl)

timeBin = 8

layer_name = 'photoreceptor_dart_multichan'
weights_layer = get_weightsOfLayer(weights_dict,layer_name)

alpha =  tf.constant(weights_layer['alpha'] / timeBin)
beta = tf.constant(weights_layer['beta'] / timeBin)
gamma =  tf.constant(weights_layer['gamma'])
tau_y =  tf.constant((weights_layer['tauY']*weights_layer['tauY_mulFac']) / timeBin)
tau_z =  tf.constant((weights_layer['tauZ']*weights_layer['tauZ_mulFac']) / timeBin)
n_y =  tf.constant((weights_layer['nY']*weights_layer['nY_mulFac']))
n_z =  tf.constant((weights_layer['nZ']*weights_layer['nZ_mulFac']))
tau_r = tf.constant((weights_layer['tauR']*weights_layer['tauR_mulFac']) / timeBin)

t = tf.range(0,1000/timeBin,dtype='float32')

Ky = generate_simple_filter_multichan(tau_y[None,:],n_y[None,:],t)
Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter_multichan(tau_z[None,:],n_z[None,:],t))

mdl_layers = [layer.name for layer in mdl.layers]
rgb = re.compile(layer_name+'*')
layer_nameAsItAppears = list(filter(rgb.match, mdl_layers))[0]
idx_layer_start = [i for i in range(len(mdl_layers)) if mdl_layers[i]==layer_nameAsItAppears][0]

x = Input(shape=data_val.X.shape[1:])
n_cells = data_val.y.shape[1]
y = x 
for layer in mdl.layers[1:idx_layer_start]:
    y = layer(y)
# y = y[:,60:,0,:,:]
# y = BatchNormalization(axis=2)(y)
# y = LayerNormalization(axis=3)(y)

mdl_d2 = Model(x, y, name='new_mdl')
obs_rate = data_val.y
data_inputToLayer = mdl_d2.predict(data_val.X)
# plt.plot(data_inputToLayer[0,:,0,0])


y_tf = conv_oper_multichan(data_inputToLayer,Ky)
z_tf = conv_oper_multichan(data_inputToLayer,Kz)

y_tf_reshape = tf.squeeze(tf.reshape(y_tf,(-1,y_tf.shape[1],y_tf.shape[2],data_inputToLayer.shape[-1],alpha.shape[-1])))
z_tf_reshape = tf.squeeze(tf.reshape(z_tf,(-1,z_tf.shape[1],z_tf.shape[2],data_inputToLayer.shape[-1],alpha.shape[-1])))

a = alpha[None,None,None,:]*y_tf_reshape
b = beta[None,None,None,:]*z_tf_reshape
c = 1+b
outputs = a/c

Ky = np.squeeze(Ky.eval(session=tf.compat.v1.Session()))
Kz = np.squeeze(Kz.eval(session=tf.compat.v1.Session()))
if Ky.ndim==1:
    Ky = Ky[:,np.newaxis]
    Kz = Kz[:,np.newaxis]

y_tf = np.squeeze(y_tf_reshape.eval(session=tf.compat.v1.Session()))
z_tf = np.squeeze(z_tf_reshape.eval(session=tf.compat.v1.Session()))

a = np.squeeze(a.eval(session=tf.compat.v1.Session()))
b = np.squeeze(b.eval(session=tf.compat.v1.Session()))
c = np.squeeze(c.eval(session=tf.compat.v1.Session()))
outputs = np.squeeze(outputs.eval(session=tf.compat.v1.Session()))

a_norm = (a-a.min())/(a.max()-a.min()); a_norm = a_norm-a_norm.mean()
c_norm = (c-c.min())/(c.max()-c.min()); c_norm = c_norm-c_norm.mean()

# outputs_norm = (outputs-outputs.min())/(outputs.max()-outputs.min())
# outputs_norm = outputs_norm-outputs_norm.mean()
if outputs.ndim==3:
    outputs = outputs[:,:,:,np.newaxis]
    
outputs_perChan = outputs.reshape(-1,outputs.shape[-1])
outputs_norm = outputs[:,outputs.shape[1]-select_T:,:,:]
outputs_norm = (outputs_norm-np.min(outputs_perChan,axis=0)[None,None,None,:])/((np.max(outputs_perChan,axis=0)[None,None,None,:]-np.min(outputs_perChan,axis=0)[None,None,None,:])+1e-5)
outputs_norm_mean = np.mean(outputs_norm.reshape(-1,outputs_norm.shape[-1]),axis=0)
outputs_norm = outputs_norm-outputs_norm_mean[None,None,None,:]

data_inputToLayer_norm = (data_inputToLayer-data_inputToLayer.min())/(data_inputToLayer.max()-data_inputToLayer.min())
data_inputToLayer_norm = data_inputToLayer_norm-data_inputToLayer_norm.mean()


font_size_ticks = 10
fig_rows = outputs_norm.shape[-1]+1
fig_cols = 2
t_frame = 8
t_axis = np.flip(-1*np.arange(0,pr_temporal_width*t_frame,t_frame))

fig_temp,axs_temp = plt.subplots(fig_rows,fig_cols,figsize=(20,12))
fig_temp.suptitle('Layer: '+layer_name+' | Filters',fontsize=16)
axs_temp = np.ravel(axs_temp)
t_axis = np.flip(-1*np.arange(0,Ky.shape[0]*t_frame,t_frame))


idx_plots_stim = 1
idx_plots_filts = np.arange(0,outputs_norm.shape[-1]*2,2)+2
idx_plots_outputs = np.arange(0,outputs_norm.shape[-1]*2,2)+3

axs_temp[idx_plots_stim].plot(data_inputToLayer_norm[0,outputs.shape[1]-select_T:,0],label='stimulus')
axs_temp[idx_plots_stim].legend()
for i in range(Ky.shape[-1]):
    axs_temp[idx_plots_filts[i]].plot(t_axis,np.flip(Ky[:,i]))
    axs_temp[idx_plots_filts[i]].plot(t_axis,np.flip(Kz[:,i]))
    
    # axs_temp[idx_plots_outputs[i]].plot(np.flip(a_norm[0,outputs.shape[1]-select_T:,0,i]),label='alpha*y')
    # axs_temp[idx_plots_outputs[i]].plot(np.flip(c_norm[0,outputs.shape[1]-select_T:,0,i]),label='1+beta*z')
    axs_temp[idx_plots_outputs[i]].plot(outputs_norm[0,:,0,i],label='alpha*y/1+beta*z')
    # axs_temp[idx_plots_outputs[i]].plot(outputs_norm[0,:,0,i],label='alpha*y/1+beta*z')
    axs_temp[idx_plots_outputs[i]].legend()
    
# %%

def generate_simple_filter(tau,n,t):
    f = (t**n)*tf.math.exp(-t/tau) # functional form in paper
    f = (f/tau**(n+1))/tf.math.exp(tf.math.lgamma(n+1)) # normalize appropriately
    return f

idx = 0
t_y = ((weights_layer['tauY']*weights_layer['tauY_mulFac']) / timeBin)[idx]
t_z = ((weights_layer['tauZ']*weights_layer['tauZ_mulFac']) / timeBin)[idx]
n_y = ((weights_layer['nY']*weights_layer['nY_mulFac']) / timeBin)[idx]
n_z = ((weights_layer['nZ']*weights_layer['nZ_mulFac']) / timeBin)[idx]
alpha = (weights_layer['alpha'] / timeBin)[idx]
beta = (weights_layer['beta'] / timeBin)[idx]

t = np.arange(0,1000/timeBin)
Ky_sing = generate_simple_filter(t_y,n_y,t)
Ky_sing = np.squeeze(Ky_sing.eval(session=tf.compat.v1.Session()))
Ky_sing = Ky_sing*alpha

Kz_sing = generate_simple_filter(t_z,n_z,t)
Kz_sing = np.squeeze(Kz_sing.eval(session=tf.compat.v1.Session()))
Kz_sing = Kz_sing*beta

plt.plot(Ky_sing)
plt.plot(Kz_sing)

# %% find how many epochs are required
select_exp = 'retina1'
select_mdl = 'CNN_2D'
for i in paramNames_unique:#[800:1000]:
    idx_bestTrial = np.argmax(perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
    fev_allEpochs = perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial]
    
    plt.plot(fev_allEpochs)
plt.show()
    


# %% correlation
def cross_corr(y1, y2):
  """Calculates the cross correlation and lags without normalization.

  The definition of the discrete cross-correlation is in:
  https://www.mathworks.com/help/matlab/ref/xcorr.html

  Args:
    y1, y2: Should have the same length.

  Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
  """
  if len(y1) != len(y2):
    raise ValueError('The lengths of the inputs should be the same.')

  y1_auto_corr = np.dot(y1, y1) / len(y1)
  y2_auto_corr = np.dot(y2, y2) / len(y1)
  corr = np.correlate(y1, y2, mode='same')
  # The unbiased sample size is N - lag.
  unbiased_sample_size = np.correlate(
      np.ones(len(y1)), np.ones(len(y1)), mode='same')
  corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
  shift = len(y1) // 2

  max_corr = np.max(corr)
  argmax_corr = np.argmax(corr)
  return max_corr, argmax_corr - shift


# % cross correlation

obs_rate = data_val.y[100:,:]
pred_rate = mdl.predict(data_val.X)[100:,:]

lags = np.empty(obs_rate.shape[1])

for i in range(obs_rate.shape[1]):
    a = obs_rate[:,i]
    b = pred_rate[:,i]
    
    corr,lags[i] = cross_corr(a, b)

lags_median = np.median(lags)
print(lags_median)
# lags = lags+2
# lags[lags>7]=7
# lags[lags<0]=7

# %%3D plot
# fev_surfacePlot = np.zeros((len(eval('select_'+select_param_x)),len(eval('select_'+select_param_y)),len(eval('select_'+select_param_z))))
# fev_surfacePlot[:] = np.nan

# totalCombs = sum(idx_interest)
# idx_interest = np.where(idx_interest)[0]

# for i in idx_interest:
#     idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
#     idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
#     idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
#     paramName = paramNames_allExps[select_exp][select_mdl][i]
#     if perf_allExps[select_exp][select_mdl][paramName]['model_performance']['num_trials'] == 1:
#         rgb = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr']
#     else:
#         rgb = np.mean(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
#     rgb = np.nanmedian(rgb,axis=0)
    
#     if rgb>.85:#thresh_fev:
#         fev_surfacePlot[idx_x,idx_y,idx_z] = rgb

# fig,axs = plt.subplots(1,1,subplot_kw={"projection": "3d"})

# X,Y,Z = np.meshgrid(eval('select_'+select_param_x),eval('select_'+select_param_y),eval('select_'+select_param_z))

# color_map = plt.cm.get_cmap('hot')
# reversed_color_map = color_map.reversed()
# surf = axs.scatter3D(X,Y,Z,c=fev_surfacePlot,cmap = reversed_color_map,linewidth=0,antialiased=False)
# axs.set_xlabel(select_param_x)
# axs.set_ylabel(select_param_y)
# axs.set_zlabel(select_param_z)
# axs.set_title(select_mdl)
# fig.colorbar(surf)
# axs.view_init(10, 320)
# plt.draw()

# %% Bar graph all models


def perf_bar(select_exp,select_mdl,mdl,val_dataset,select_T):
    fname_data_train_val_test = os.path.join(path_dataset,(select_exp+'_dataset_train_val_test_'+val_dataset+'.h5'))
    _,data_val,_,_,dataset_rr,_,resp_orig = load_h5Dataset(fname_data_train_val_test)
    
    resp_orig = resp_orig['train']
    
    
    if select_mdl[:6]=='CNN_2D':
        data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
    elif select_mdl[:6]=='CNN_3D':
        data_val = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))
        
    elif select_mdl[:6]=='convLS' or select_mdl == 'LSTM_CNN_2D':
        data_val = prepare_data_convLSTM(data_val,select_T,np.arange(data_val.y.shape[1]))
        
    
    filt_temporal_width = select_T
    obs_rate_allStimTrials_d1 = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]
    obs_rate = data_val.y
    
    if correctMedian==True:
        fname_data_train_val_test_d1 = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_train+'.h5'))
        _,_,_,_,_,_,resp_med_d1 = load_h5Dataset(fname_data_train_val_test_d1)
        resp_med_d1 = np.nanmedian(resp_med_d1['train'],axis=0)
        resp_med_d2 = np.nanmedian(resp_orig,axis=0)
        resp_mulFac = resp_med_d2/resp_med_d1;
        
        pred_rate = mdl.predict(data_val.X)
        pred_rate = pred_rate * resp_mulFac[None,:]
    
        
    else:       
        pred_rate = mdl.predict(data_val.X)
    _ = gc.collect()
    tf.keras.backend.clear_session()
    # obs_rate = data_test.y
    # pred_rate = mdl.predict(data_test.X)
    
    
    num_iters = 50
    fev_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
    fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
    predCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
    rrCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
    
    for i in range(num_iters):
        fev_d1_allUnits[:,i], fracExplainableVar[:,i], predCorr_d1_allUnits[:,i], rrCorr_d1_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_d1,pred_rate,0,RR_ONLY=False,lag = samps_shift)
    
    
    fev_d1_allUnits = np.mean(fev_d1_allUnits,axis=1)
    fracExplainableVar = np.mean(fracExplainableVar,axis=1)
    predCorr_d1_allUnits = np.mean(predCorr_d1_allUnits,axis=1)
    rrCorr_d1_allUnits = np.mean(rrCorr_d1_allUnits,axis=1)
    
    idx_allUnits = np.arange(fev_d1_allUnits.shape[0])
    idx_d1_valid = idx_allUnits
    idx_d1_valid = np.logical_and(fev_d1_allUnits>-1,fev_d1_allUnits<1.1)
    idx_d1_valid = idx_allUnits[idx_d1_valid]
    
    
    fev_d1_medianUnits = np.median(fev_d1_allUnits[idx_d1_valid])
    fev_d1_stdUnits = np.std(fev_d1_allUnits[idx_d1_valid])
    fev_d1_ci = 1.96*(fev_d1_stdUnits/len(idx_d1_valid)**.5)
    
    predCorr_d1_medianUnits = np.median(predCorr_d1_allUnits[idx_d1_valid])
    predCorr_d1_stdUnits = np.std(predCorr_d1_allUnits[idx_d1_valid])
    predCorr_d1_ci = 1.96*(predCorr_d1_stdUnits/len(idx_d1_valid)**.5)
    
    rrCorr_d1_medianUnits = np.median(rrCorr_d1_allUnits[idx_d1_valid])
    rrCorr_d1_stdUnits = np.std(rrCorr_d1_allUnits[idx_d1_valid])
    rrCorr_d1_ci = 1.96*(rrCorr_d1_stdUnits/len(idx_d1_valid)**.5)
    
    
    
    return fev_d1_allUnits,fev_d1_medianUnits, fev_d1_ci, predCorr_d1_allUnits, predCorr_d1_medianUnits, predCorr_d1_ci, rrCorr_d1_medianUnits, rrCorr_d1_ci, idx_d1_valid 

select_exp = expDates[0]
select_mdl = models_all[0] #'CNN_2D' #'CNN_2D_chansVary'#'CNN_2D_filtsVary'

val_dataset_train= lightLevel_1      # ['scotopic','photopic']
val_dataset_test = lightLevel_1
val_dataset_test_2 = 'scotopic' #'scotopic' #'scotopic-1_preproc-added_norm-1_rfac-2' 

if len(val_dataset_train) == 8 and len(val_dataset_test_2)!=8:
    raise ValueError('wrong datasets')

if len(val_dataset_train) > 8 and len(val_dataset_test_2)==8:
    raise ValueError('wrong datasets')    
    
correctMedian = False

fname_data_train_val_test = os.path.join(path_dataset,(select_exp+'_dataset_train_val_test_'+val_dataset_test+'.h5'))
_,_,_,_,_,parameters,_ = load_h5Dataset(fname_data_train_val_test)
samps_shift = int(np.array(parameters['samps_shift']))

fev_d1_medianUnits_allMdls = np.atleast_1d([])
fev_d1_ci_allMdls = np.atleast_1d([])
predCorr_d1_medianUnits_allMdls = np.atleast_1d([])
predCorr_d1_ci_allMdls = np.atleast_1d([])

fev_d2_medianUnits_allMdls = np.atleast_1d([])
fev_d2_ci_allMdls = np.atleast_1d([])
predCorr_d2_medianUnits_allMdls = np.atleast_1d([])
predCorr_d2_ci_allMdls = np.atleast_1d([])

trainingSamps_allMdls = np.atleast_1d([])

mdl_name_params = []
item_xlabel = 'BN'


for select_mdl in models_all:

    paramFileName = list(perf_allExps[select_exp][select_mdl].keys())   
    
    for p in range(0,len(paramFileName)):
        
        print('Evaluating %d of %d' %(p+1,len(paramFileName)))
    
        idx_bestTrial = np.nanargmax(perf_allExps[select_exp][select_mdl][paramFileName[p]]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
        idx_bestEpoch = np.nanargmax(perf_allExps[select_exp][select_mdl][paramFileName[p]]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])
        # plt.plot(perf_allExps[select_exp][select_mdl][paramFileName[p]]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])
        
        select_TR = idx_bestTrial+1
        
        # resp_median_photopic = perf_allExps[select_exp][select_mdl][paramFileName]['dataset_pred']['resp_median_allUnits']
        mdlFolder = paramFileName[p]+'_TR-%02d' % select_TR
        
        path_model = os.path.join(path_mdl_drive,select_mdl,mdlFolder)
        mdl = load(os.path.join(path_model,mdlFolder))
    
        select_T = perf_allExps[select_exp][select_mdl][paramFileName[p]]['model_params']['temporal_width']
        
        fev_d1_allUnits,fev_d1_medianUnits, fev_d1_ci, predCorr_d1_allUnits, predCorr_d1_medianUnits, predCorr_d1_ci, rrCorr_d1_medianUnits, rrCorr_d1_ci, idx_d1_valid = perf_bar(select_exp,select_mdl,mdl,val_dataset_test,select_T)

        fev_d1_medianUnits_allMdls = np.append(fev_d1_medianUnits_allMdls,fev_d1_medianUnits)
        fev_d1_ci_allMdls = np.append(fev_d1_ci_allMdls,fev_d1_ci)
        predCorr_d1_medianUnits_allMdls = np.append(predCorr_d1_medianUnits_allMdls,predCorr_d1_medianUnits)
        predCorr_d1_ci_allMdls = np.append(predCorr_d1_ci_allMdls,predCorr_d1_ci)
        
        if val_dataset_test_2 != 0:
            fev_d2_allUnits,fev_d2_medianUnits, fev_d2_ci, predCorr_d2_allUnits, predCorr_d2_medianUnits, predCorr_d2_ci, rrCorr_d2_medianUnits, rrCorr_d2_ci, idx_d2_valid = perf_bar(select_exp,select_mdl,mdl,val_dataset_test_2,select_T)

            fev_d2_medianUnits_allMdls = np.append(fev_d2_medianUnits_allMdls,fev_d2_medianUnits)
            fev_d2_ci_allMdls = np.append(fev_d2_ci_allMdls,fev_d2_ci)
            predCorr_d2_medianUnits_allMdls = np.append(predCorr_d2_medianUnits_allMdls,predCorr_d2_medianUnits)
            predCorr_d2_ci_allMdls = np.append(predCorr_d2_ci_allMdls,predCorr_d2_ci)


        trainingSamps_allMdls = np.append(trainingSamps_allMdls,int(paramFileName[p][-2:]))

        # rgb = select_mdl#+'_'+paramFileName[p][13:-10]
        rgb = item_xlabel+'-'+str(params_allExps[select_exp][select_mdl][item_xlabel][p])
        mdl_name_params.append(rgb)
        
# %        
fig,axs = plt.subplots(2,1,figsize=(5,10))
fig.suptitle(select_exp+'\nTraining: '+val_dataset_train,fontsize=16)
font_size_ticks = 16
font_size_labels = 16


col_scheme = ('darkgrey',cols_lightLevels[val_dataset_test[:8]])
# ax.yaxis.grid(True)
xpoints = np.arange(fev_d1_medianUnits_allMdls.shape[0])
xlabel_fev = mdl_name_params
axs[0].bar(xpoints-.2,fev_d1_medianUnits_allMdls*100,yerr=fev_d1_ci_allMdls*100,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test[:8]],width=0.4,label=val_dataset_test[:8])
axs[0].bar(xpoints+.2,fev_d2_medianUnits_allMdls*100,yerr=fev_d2_ci_allMdls*100,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test_2[:8]],width=0.4,label=val_dataset_test_2[:8])
axs[0].bar(len(xpoints)-.2,0,yerr=0,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test[:8]],width=0.4)
axs[0].bar(len(xpoints)+.2,0,yerr=0,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test_2[:8]],width=0.4)

axs[0].set_xticks(xpoints)#(2*np.arange(0,fev_d1_medianUnits_allMdls.shape[0]))
axs[0].set_xticklabels(xlabel_fev)
axs[0].set_yticks(np.arange(-1,1.1,.1)*100)
axs[0].set_ylabel('FEV',fontsize=font_size_ticks)
axs[0].set_title('',fontsize=font_size_ticks)
axs[0].set_ylim((0,101))
axs[0].tick_params(axis='both',labelsize=16)
# axs[0].plot((-0.5,len(mdl_name_params)-0.5),(np.nanmax(fev_d1_medianUnits_allMdls),np.nanmax(fev_d1_medianUnits_allMdls)),color='green')
# axs[0].legend(loc='best',fontsize=font_size_labels)


axs[1].bar(xpoints-.2,predCorr_d1_medianUnits_allMdls,yerr=predCorr_d1_ci_allMdls,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test[:8]],width=0.4,label=val_dataset_test[:8])
axs[1].bar(xpoints+.2,predCorr_d2_medianUnits_allMdls,yerr=predCorr_d2_ci_allMdls,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test_2[:8]],width=0.4,label=val_dataset_test_2[:8])
axs[1].bar(len(xpoints)-.2,rrCorr_d1_medianUnits,yerr=rrCorr_d1_ci,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test[:8]],width=0.4)
axs[1].bar(len(xpoints)+.2,rrCorr_d2_medianUnits,yerr=rrCorr_d2_ci,align='center',capsize=6,alpha=.7,color=cols_lightLevels[val_dataset_test_2[:8]],width=0.4)
xpoints = np.arange(fev_d1_medianUnits_allMdls.shape[0]+1)
xlabel_corr = list(mdl_name_params)
xlabel_corr.append('RetinaReliab')
axs[1].set_xticks(xpoints)#(2*np.arange(0,fev_d1_medianUnits_allMdls.shape[0]))
axs[1].set_xticklabels(xlabel_corr)
axs[1].set_yticks(np.arange(-0.6,1.1,.2))
axs[1].set_ylabel('Correlation Coefficient',fontsize=font_size_ticks)
axs[1].set_title('',fontsize=font_size_ticks)
axs[1].set_ylim((0,1.1))
axs[1].tick_params(axis='both',labelsize=16)
# axs[1].plot((-0.5,len(mdl_name_params)),(np.nanmax(predCorr_d1_medianUnits_allMdls),np.nanmax(predCorr_d1_medianUnits_allMdls)),color='green')
# axs[1].legend(loc='best',fontsize=font_size_labels)


