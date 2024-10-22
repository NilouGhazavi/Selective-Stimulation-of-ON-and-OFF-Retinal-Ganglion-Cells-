#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:25:03 2023

@author: saad
"""

path_figs = '/home/saad/postdoc_db/papers/PR_paper/figs/'
plt.rcParams['svg.fonttype'] = 'none'

import seaborn as sns
import pandas as pd
import scipy


# %% Grads variation

# uname = on_mid_014
# light level = scot-30

fig_title = '%s | %s'%(uname,select_lightLevel)

t_frame = 8
tempRF_scaled = tempRF_grand*1e3/t_frame

# tempRF_scaled = tempRF_scaled[:,::dsFac]

x_ticksLabel = -t_frame*np.arange(tempRF_scaled.shape[-1])
x_ticksLabel = np.flip(x_ticksLabel)
x_ticks = np.arange(tempRF_scaled.shape[-1])[::5]
x_ticksLabel = x_ticksLabel[::5]


fig,axs = plt.subplots(1,1,figsize=(7,5))
axs = np.ravel(axs)

axs[0].plot(tempRF_scaled[:100,:].T)
axs[0].set_xticks(x_ticks)
axs[0].set_xticklabels(x_ticksLabel)
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('spikes/R*/rod/s')
axs[0].set_title(fig_title)

# fig_name = 'tempRF_grads'
# fig.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
# fig.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)


# Histogram
peakvals = np.max(tempRF_scaled,axis=1)
bins = np.arange(0,1,0.01)

fig,axs = plt.subplots(1,1,figsize=(7,5))
axs = np.ravel(axs)
axs[0].hist(peakvals,bins)
axs[0].set_xlim([0.000,0.7])
axs[0].set_ylim([0,20000])
axs[0].set_title(fig_title)
axs[0].set_xlabel('spikes/R*/rod/s')
axs[0].set_ylabel('N_samps | max:'+str(len(peakvals)))

# fig_name = 'tempRF_hist'
# fig.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
# fig.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)


# %% Example cell binned - 30Rstar
"""
on_mid_029 is a very good example for 0.3 rstar
on_mid_14[11] and on_mid_16[13] are good examples for 3R*
"""

fname_gainFile = '/home/saad/postdoc_db/analyses/data_kiersten/monkey01/gradient_analysis/gain_analysis_ds.h5'
f = h5py.File(fname_gainFile,'r')

select_mdl = 'PRFR_CNN2D_RODS'  # CNN_2D_NORM # PRFR_CNN2D_RODS
select_lightLevel = 'scot-30-Rstar'

uname_gainFile = list(f[select_mdl][select_lightLevel].keys()) #['on_mid_003', 'on_mid_004', 'on_mid_005', 'on_mid_006', 'on_mid_009', 'on_mid_011', 'on_mid_015', 'on_mid_016', 'on_mid_017', 'on_mid_018', 'on_mid_020']
temp_win = 40
nbins = 10
dsFac = 4
temp_win_ds = int(temp_win/dsFac)

gain_grads_cnn = np.zeros((nbins,len(uname_gainFile)));gain_grads_cnn[:]=np.nan
gain_real_cnn = np.zeros((nbins,len(uname_gainFile)));gain_real_cnn[:]=np.nan
tempRF_grads_cnn = np.zeros((temp_win,nbins,len(uname_gainFile)));tempRF_grads_cnn[:]=np.nan
tempRF_real_cnn = np.zeros((temp_win_ds,nbins,len(uname_gainFile)));tempRF_real_cnn[:]=np.nan
fevs_cnn = np.zeros((len(uname_gainFile)));fevs_cnn[:]=np.nan

gain_grads_pr = np.zeros((nbins,len(uname_gainFile)));gain_grads_cnn[:]=np.nan
gain_real_pr = np.zeros((nbins,len(uname_gainFile)));gain_real_cnn[:]=np.nan
tempRF_grads_pr = np.zeros((temp_win,nbins,len(uname_gainFile)));tempRF_grads_cnn[:]=np.nan
tempRF_real_pr = np.zeros((temp_win_ds,nbins,len(uname_gainFile)));tempRF_real_cnn[:]=np.nan
fevs_pr = np.zeros((len(uname_gainFile)));fevs_pr[:]=np.nan


for u in range(len(uname_gainFile)):
    uname = uname_gainFile[u]
    gain_grads_cnn[:,u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['gain_grads_binned'])
    gain_real_cnn[:,u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['gain_real_binned'])
    tempRF_grads_cnn[:,:,u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['tempRF_grads_binned_grand'][-temp_win:])
    tempRF_real_cnn[:,:,u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['tempRF_real_binned_grand'][-temp_win_ds:])
    fevs_cnn[u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['fev'])
    
    gain_grads_pr[:,u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['gain_grads_binned'])
    gain_real_pr[:,u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['gain_real_binned'])
    tempRF_grads_pr[:,:,u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['tempRF_grads_binned_grand'][-temp_win:])
    tempRF_real_pr[:,:,u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['tempRF_real_binned_grand'][-temp_win_ds:])
    fevs_pr[u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['fev'])
    
f.close()
    # plt.plot(gain_grads,gain_real,'o');plt.ylabel('real');plt.xlabel('grads');plt.title(slect_mdl+' | '+select_lightLevel+' | '+uname);plt.show()

# tempRF_grads_cnn = tempRF_grads_cnn[::dsFac]
# tempRF_grads_pr = tempRF_grads_pr[::dsFac]

binsToTake = np.array([0,1,2,3,4,5,6,7,8,9])

mse_cnn = np.nanmean((gain_grads_cnn[binsToTake]-gain_real_cnn[binsToTake])**2,axis=0)
mse_pr = np.nanmean((gain_grads_pr[binsToTake]-gain_real_pr[binsToTake])**2,axis=0)

idx_fev_CNN_G_PR = fevs_cnn>=fevs_pr
idx_fev_PR_G_CNN = ~idx_fev_CNN_G_PR

# max_axis = np.nanmax((np.nanmax(mse_cnn),np.nanmax(mse_pr)))+.01
# plt.plot(mse_cnn[idx_fev_PR_G_CNN],mse_pr[idx_fev_PR_G_CNN],'ro',label='PR>CNN');plt.plot(mse_cnn[idx_fev_CNN_G_PR],mse_pr[idx_fev_CNN_G_PR],'bo',label='CNN>PR')
# plt.plot([0,1],[0,1],'--k'); plt.xlim(0,max_axis); plt.ylim(0,max_axis);plt.show()

tempRF_grads_pr_newNorm = tempRF_grads_pr/np.nanmax(tempRF_grads_pr[:,binsToTake,:],axis=(0,1),keepdims=True)
tempRF_real_pr_newNorm = tempRF_real_pr/np.nanmax(tempRF_real_pr[:,binsToTake,:],axis=(0,1),keepdims=True)

tempRF_grads_cnn_newNorm = tempRF_grads_cnn/np.nanmax(tempRF_grads_cnn[:,binsToTake,:],axis=(0,1),keepdims=True)
tempRF_real_cnn_newNorm = tempRF_real_cnn/np.nanmax(tempRF_real_cnn[:,binsToTake,:],axis=(0,1),keepdims=True)

t_frame = 8
t_frame_ds = int(t_frame*dsFac)
x_ticksLabel = -t_frame*np.arange(tempRF_grads_cnn.shape[0])
x_ticksLabel = np.flip(x_ticksLabel)[::5]
x_ticks = np.arange(tempRF_grads_cnn.shape[0])[::5]

x_ticksLabel_ds = -t_frame_ds*np.arange(tempRF_real_cnn.shape[0])
x_ticksLabel_ds = np.flip(x_ticksLabel_ds)
x_ticks_ds = np.arange(tempRF_real_cnn.shape[0])

# on_mid_014(11) on_mid_016 (13)
u_idx = 13
# for u_idx in range(mse_pr.shape[0]):
fig_title = '%s | %s'%(uname_gainFile[u_idx],select_lightLevel)
fig,axs = plt.subplots(2,2,figsize=(20,10))
axs = np.ravel(axs)
fig.suptitle(fig_title)
axs[0].plot(tempRF_grads_pr_newNorm[:,binsToTake,u_idx])
axs[0].set_xticks(x_ticks)
axs[0].set_xticklabels(x_ticksLabel)
axs[0].set_ylim([-0.1,1.1])
axs[1].plot(tempRF_real_pr_newNorm[:,binsToTake,u_idx])
axs[1].set_xticks(x_ticks_ds)
axs[1].set_xticklabels(x_ticksLabel_ds)
axs[1].set_title('%s | FEV = %02d%%'%('PRFR',fevs_pr[u_idx]))
axs[1].set_ylim([-0.1,1.1])

axs[2].plot(tempRF_grads_cnn_newNorm[:,binsToTake,u_idx])
axs[2].set_xticks(x_ticks)
axs[2].set_xticklabels(x_ticksLabel)
axs[3].set_ylim([-0.1,1.1])
axs[3].plot(tempRF_real_cnn_newNorm[:,binsToTake,u_idx])
axs[3].set_xticks(x_ticks_ds)
axs[3].set_xticklabels(x_ticksLabel_ds)
axs[3].set_title('%s | FEV = %02d%%'%('CNN',fevs_cnn[u_idx]))
axs[3].set_ylim([-0.1,1.1])
plt.show()


# fig_name = 'tempRF_binned'
# fig.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
# fig.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)

# binsToTake = np.array([5,6,7,8,9])
# plt.plot(np.nanmean(tempRF_grads_pr_newNorm[:,binsToTake,u_idx],axis=1))
# plt.plot(np.nanmean(tempRF_grads_cnn_newNorm[:,binsToTake,u_idx],axis=1))
# plt.show()
# %% scatter plots MSE
import sklearn
fname_gainFile = '/home/saad/postdoc_db/analyses/data_kiersten/monkey01/gradient_analysis/gain_analysis_ds.h5'
f = h5py.File(fname_gainFile,'r')

select_mdl = 'PRFR_CNN2D_RODS'  # CNN_2D_NORM # PRFR_CNN2D_RODS
lightLevels_all = ('scot-30-Rstar','scot-3-Rstar','scot-0.3-Rstar')

temp_win = 40
dsFac
temp_win_ds = int(temp_win/dsFac)
nbins = 10
    
# plt.hist(lat_pr_diff[:,0])
mean_mse_cnn = np.zeros(len(lightLevels_all))
mean_mse_pr = np.zeros(len(lightLevels_all))
p_mse = np.zeros(len(lightLevels_all))
slope_mse = np.zeros(len(lightLevels_all))


fig,axs = plt.subplots(1,3,figsize=(20,5))
axs = np.ravel(axs)
ctr_axis = -1
for select_lightLevel in lightLevels_all:
    ctr_axis += 1
    uname_gainFile = list(f[select_mdl][select_lightLevel].keys()) #['on_mid_003', 'on_mid_004', 'on_mid_005', 'on_mid_006', 'on_mid_009', 'on_mid_011', 'on_mid_015', 'on_mid_016', 'on_mid_017', 'on_mid_018', 'on_mid_020']
    
    gain_grads_cnn = np.zeros((nbins,len(uname_gainFile)));gain_grads_cnn[:]=np.nan
    gain_real_cnn = np.zeros((nbins,len(uname_gainFile)));gain_real_cnn[:]=np.nan
    tempRF_grads_cnn = np.zeros((temp_win,nbins,len(uname_gainFile)));tempRF_grads_cnn[:]=np.nan
    tempRF_real_cnn = np.zeros((temp_win_ds,nbins,len(uname_gainFile)));tempRF_real_cnn[:]=np.nan
    fevs_cnn = np.zeros((len(uname_gainFile)));fevs_cnn[:]=np.nan
    
    gain_grads_pr = np.zeros((nbins,len(uname_gainFile)));gain_grads_cnn[:]=np.nan
    gain_real_pr = np.zeros((nbins,len(uname_gainFile)));gain_real_cnn[:]=np.nan
    tempRF_grads_pr = np.zeros((temp_win,nbins,len(uname_gainFile)));tempRF_grads_cnn[:]=np.nan
    tempRF_real_pr = np.zeros((temp_win_ds,nbins,len(uname_gainFile)));tempRF_real_cnn[:]=np.nan
    fevs_pr = np.zeros((len(uname_gainFile)));fevs_pr[:]=np.nan
    
    
    for u in range(len(uname_gainFile)):
        uname = uname_gainFile[u]
        gain_grads_cnn[:,u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['gain_grads_binned'])
        gain_real_cnn[:,u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['gain_real_binned'])
        tempRF_grads_cnn[:,:,u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['tempRF_grads_binned_grand_norm'][-temp_win:])
        tempRF_real_cnn[:,:,u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['tempRF_real_binned_grand_norm'][-temp_win_ds:])
        fevs_cnn[u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['fev'])
        
        gain_grads_pr[:,u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['gain_grads_binned'])
        gain_real_pr[:,u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['gain_real_binned'])
        tempRF_grads_pr[:,:,u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['tempRF_grads_binned_grand_norm'][-temp_win:])
        tempRF_real_pr[:,:,u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['tempRF_real_binned_grand_norm'][-temp_win_ds:])
        fevs_pr[u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['fev'])
        
        # plt.plot(gain_grads,gain_real,'o');plt.ylabel('real');plt.xlabel('grads');plt.title(slect_mdl+' | '+select_lightLevel+' | '+uname);plt.show()
    
    binsToTake = np.array([0,1,2,3,4,5,6,7,8,9])
    gain_grads_cnn = gain_grads_cnn/np.nanmax(gain_grads_cnn[binsToTake],keepdims=True)
    gain_grads_pr = gain_grads_pr/np.nanmax(gain_grads_pr[binsToTake],keepdims=True)
    gain_real_cnn = gain_real_cnn/np.nanmax(gain_real_cnn[binsToTake],keepdims=True)
    gain_real_pr = gain_real_pr/np.nanmax(gain_real_pr[binsToTake],keepdims=True)

    mse_cnn = np.nanmean((gain_grads_cnn[binsToTake]-gain_real_cnn[binsToTake])**2,axis=0)
    mse_pr = np.nanmean((gain_grads_pr[binsToTake]-gain_real_pr[binsToTake])**2,axis=0)
    
    mean_mse_cnn[ctr_axis] = np.nanmean(mse_cnn)
    mean_mse_pr[ctr_axis] = np.nanmean(mse_pr)
    _,p_mse[ctr_axis] = scipy.stats.wilcoxon(mse_pr,mse_cnn,alternative='less')
    


    idx_fev_CNN_G_PR = fevs_cnn>=fevs_pr
    idx_fev_PR_G_CNN = ~idx_fev_CNN_G_PR
    
    max_axis = np.max((mse_cnn.max(),mse_pr.max()))+.02
    txt_title = 'Training: %s | Testing: %s | N=%d RGCs'%(dataset_model,select_lightLevel,len(uname_gainFile))
    
    axs[ctr_axis].plot(mse_cnn[idx_fev_PR_G_CNN],mse_pr[idx_fev_PR_G_CNN],'ro',label='PR>CNN')
    axs[ctr_axis].plot(mse_cnn[idx_fev_CNN_G_PR],mse_pr[idx_fev_CNN_G_PR],'bo',label='CNN>PR')
    axs[ctr_axis].legend()
    axs[ctr_axis].plot([0,1],[0,1],'--k')
    axs[ctr_axis].set_xlim(0,max_axis)
    axs[ctr_axis].set_ylim(0,max_axis)
    axs[ctr_axis].set_xlabel('MSE_CNN');axs[ctr_axis].set_ylabel('MSE_PR')
    axs[ctr_axis].set_title(txt_title)
    axs[ctr_axis].set_aspect('equal','box')

f.close()

# fig_name = 'mse_tempRF'
# fig.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
# fig.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)


# %% FEVs for all cells at all light levels

fname_gainFile = '/home/saad/postdoc_db/analyses/data_kiersten/monkey01/gradient_analysis/gain_analysis.h5'
f = h5py.File(fname_gainFile,'r')

select_mdl = 'PRFR_CNN2D_RODS'  # CNN_2D_NORM # PRFR_CNN2D_RODS
lightLevels_all = ('scot-30-Rstar','scot-3-Rstar','scot-0.3-Rstar')

fevs_pr = np.zeros((len(uname_gainFile),len(lightLevels_all)));fevs_pr[:]=np.nan
fevs_cnn = np.zeros((len(uname_gainFile),len(lightLevels_all)));fevs_pr[:]=np.nan


ctr_axis = -1
for select_lightLevel in lightLevels_all:
    ctr_axis += 1
    uname_gainFile = list(f[select_mdl][select_lightLevel].keys()) #['on_mid_003', 'on_mid_004', 'on_mid_005', 'on_mid_006', 'on_mid_009', 'on_mid_011', 'on_mid_015', 'on_mid_016', 'on_mid_017', 'on_mid_018', 'on_mid_020']
    temp_win = 40
    nbins = 10
    
    
    
    for u in range(len(uname_gainFile)):
        uname = uname_gainFile[u]
        fevs_cnn[u,ctr_axis] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['fev'])
        fevs_pr[u,ctr_axis] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['fev'])

# %% Plot PR output
nsamps = 1500  #data.X.shape[0]
idx_chunk = np.arange(0,nsamps)

idx_unitsToExtract = np.array([2,3,4,5,8])
n_units = len(idx_unitsToExtract)

endLayer_pr = 5        # 5 includes layer norm
mdl_photocurrs = tf.keras.models.Model(inputs=mdl_dict['PRFR_CNN2D_RODS'].input,
                                       outputs=mdl_dict['PRFR_CNN2D_RODS'].layers[endLayer_pr].output)

endLayer_cnn = 1 
mdl_cnn = tf.keras.models.Model(inputs=mdl_dict['CNN_2D_NORM'].input,
                                       outputs=mdl_dict['CNN_2D_NORM'].layers[endLayer_cnn].output)

temp_width_cnn = 120
y_pr = np.zeros((len(idx_chunk),temp_width_cnn,data_alldsets['spat_dims'][0],data_alldsets['spat_dims'][1],len(dataset_eval))); y_pr[:] = np.nan
y_cnn = np.zeros((len(idx_chunk),temp_width_cnn,data_alldsets['spat_dims'][0],data_alldsets['spat_dims'][1],len(dataset_eval))); y_cnn[:] = np.nan

X_all = np.zeros((len(idx_chunk),temp_width_cnn,data_alldsets['spat_dims'][0],data_alldsets['spat_dims'][1],len(dataset_eval))); X_all[:] = np.nan
X_all_norm = np.zeros((len(idx_chunk),temp_width_cnn,data_alldsets['spat_dims'][0],data_alldsets['spat_dims'][1],len(dataset_eval))); X_all_norm[:] = np.nan

ctr_d = -1
d = dataset_eval[0]
for d in dataset_eval:
    ctr_d +=1
    data = data_alldsets[d]['raw']
       
    tempWidth_inp = mdl_dict['PRFR_CNN2D_RODS'].input.shape[1]
    X = data.X[idx_chunk][:,-tempWidth_inp:]   
    pred = mdl_photocurrs.predict(X)
    y_pr[:,:,:,:,ctr_d] = pred
    
    tempWidth_inp = mdl_dict['CNN_2D_NORM'].input.shape[1]
    X = data.X[idx_chunk][:,-tempWidth_inp:]   
    pred = mdl_cnn.predict(X)
    y_cnn[:,:,:,:,ctr_d] = pred

    X_all[:,:,:,:,ctr_d] = X
    
_ = gc.collect()

# ---- select check idx
# rgb = y_pr[:,-1,:,:,1:]
rgb = y_pr[:,-1,:,:,1]/y_pr[:,-1,:,:,2]
a = rgb
# a = rgb[:,:,:,0] - rgb[:,:,:,1]
a = np.abs(a)
b = a.reshape(a.shape[0],-1)
# b = b/np.max(b,axis=0,keepdims=True)
med = np.median(b,axis=0)
std = np.std(b,axis=0)
plt.hist(med);plt.show()
plt.hist(std);plt.show()
# idx_max = np.argmax(med)
idx_max = np.argmax(std)
idx_max_2d = np.unravel_index(idx_max,(y_pr.shape[2],y_pr.shape[3]))
plt.plot(b[:,idx_max]);plt.show()
plt.plot(y_pr[900:1100,-1,idx_max_2d[0],idx_max_2d[1],1:])

# c = np.abs(y_pr[:,-1,idx_max_2d[0],idx_max_2d[1],1] - y_pr[:,-1,idx_max_2d[0],idx_max_2d[1],2])
# plt.plot(b[:,idx_max]);plt.plot(c)

# ---- Plot PR
t_frame = 8
X_norm = (X - X.mean())/X.std()
idx_plot = 890+np.arange(0,150) #np.arange(0,len(idx_chunk) #900 1500
# idx_plot = np.concatenate((np.arange(981,1048),np.arange(1151,1200)))
x_ticks = np.arange(0,len(idx_plot),50)
x_ticks_labels = x_ticks*t_frame
idx_d = np.array([0,1,2])
idx_check = np.array([0,2])


fig,axs = plt.subplots(2,1,figsize=(20,10))
axs = np.ravel(axs)

# axs[0].plot(X_all[idx_plot,-1,idx_check[0],idx_check[1]][:,idx_d]*1e3/t_frame)
# axs[0].set_xticks(x_ticks)
# axs[0].set_xticklabels(x_ticks_labels)
# axs[0].set_xlabel('Time (ms)')
# axs[0].set_ylabel('R*/rod/s')
# axs[0].set_yscale('log')

fac = 100
stim_upsamp = np.repeat(X_all[idx_plot,-1,idx_check[0],idx_check[1]][:,idx_d],fac,axis=0)
x_ticks_upsamp = np.arange(0,len(idx_plot)*fac,50*fac)
axs[0].plot(stim_upsamp*1e3/t_frame)
axs[0].set_xticks(x_ticks_upsamp)
axs[0].set_xticklabels(x_ticks_labels)
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('R*/rod/s')
axs[0].set_yscale('log')


axs[1].plot(y_pr[idx_plot,-1,idx_check[0],idx_check[1]][:,idx_d])
# axs[1].plot(y_pr[idx_plot,-1,idx_check[0],idx_check[1]][:,idx_d[1]],'--')
axs[1].set_xticks(x_ticks)
axs[1].set_xticklabels(x_ticks_labels)
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('normalized Rstar (CNN 1st layer)')

# # axs[2].plot(X_norm[idx_plot,-1,idx_check[0],idx_check[1]],'--k')
# axs[2].plot(y_cnn[idx_plot,-1,idx_check[0],idx_check[1]][:,idx_d[0]])
# axs[2].plot(y_cnn[idx_plot,-1,idx_check[0],idx_check[1]][:,idx_d[1]],'--')
# axs[2].set_xticks(x_ticks)
# axs[2].set_xticklabels(x_ticks_labels)
# axs[2].set_xlabel('Time (ms)')
# axs[2].set_ylabel('normalized photocurrents')

fig_name = 'pr_output'
fig.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
fig.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)

# %% STA vs Grads

fname_stas =  '/home/saad/postdoc_db/analyses/data_kiersten/monkey01/db_files/datasets/monkey01_STAs_allLightLevels_8ms_Rstar.h5'
datasets_plot = ('scot-3-Rstar',)#'scot-0.3-Rstar',)#'scot-3-Rstar','scot-0.3-Rstar')
mdls_toplot = ('CNN_2D_NORM','PRFR_CNN2D_RODS',) #PRFR_CNN2D_RODS  CNN_2D_NORM

path_gradFiles = '/home/saad/data_hdd/analyses/data_kiersten/monkey01/gradient_analysis/gradients/'

temporal_width_grads = 50
temp_window = 50
sig_fac = 1.5
range_tempFilt = np.arange(temporal_width_grads-temp_window,temporal_width_grads)

u_arr = [0]

 
m = 0
num_samps = len(idx_samps) 
n_units = 7

for u in u_arr: #np.arange(0,len(perf_model['uname_selectedUnits'])):

    spatRF_sta = np.zeros((data_alldsets['spat_dims'][0],data_alldsets['spat_dims'][1],len(datasets_plot),len(mdl_names)))
    tempRF_sta = np.zeros((range_tempFilt.shape[0],len(datasets_plot),len(mdl_names)))
    spatRF_singImg = np.zeros((data_alldsets['spat_dims'][0],data_alldsets['spat_dims'][1],len(datasets_plot),len(mdl_names)))
    tempRF_singImg = np.zeros((range_tempFilt.shape[0],len(datasets_plot),len(mdl_names)))
    spatRF_gradAvg_acrossImgs = np.zeros((data_alldsets['spat_dims'][0],data_alldsets['spat_dims'][1],len(datasets_plot),len(mdl_names)))
    tempRF_gradAvg_acrossImgs = np.zeros((range_tempFilt.shape[0],len(datasets_plot),len(mdl_names)))
    spatRF_indiv_avg_acrossImgs = np.zeros((data_alldsets['spat_dims'][0],data_alldsets['spat_dims'][1],len(datasets_plot),len(mdl_names)))
    tempRF_indiv_avg_acrossImgs = np.zeros((range_tempFilt.shape[0],len(datasets_plot),len(mdl_names)))
    tempRF_indiv = np.zeros((range_tempFilt.shape[0],num_samps,len(datasets_plot),len(mdl_names)))


    for m in range(len(mdls_toplot)):
        select_mdl = mdls_toplot[m]
                
        
        ctr_d = -1
        d = datasets_plot[0]
        for d in datasets_plot:
            fname_gradsFile = os.path.join(path_gradFiles,'grads_'+select_mdl+'_'+d+'_'+str(num_samps)+'_u-'+str(n_units)+'.h5')
            f_grads = h5py.File(fname_gradsFile,'r')

            uname_all_grads = np.array(f_grads[select_mdl][d]['unames'],'bytes')
            uname_all_grads = utils_si.h5_tostring(uname_all_grads)
            uname = uname_all_grads[u]

            
            print(uname)
            select_rgc_dataset = np.where(uname==uname_all_inData)[0][0]

            ctr_d+=1
            

            data = data_alldsets[d]['raw']
        
            # Method 1: Compute STA by taking Response Weighted Average of the stimulus (model independent)
            f = h5py.File(fname_stas,'r')
            spatial_feat = np.array(f[d[:-6]][uname]['spatial_feature'])
            temporal_feat = np.array(f[d[:-6]][uname]['temporal_feature'])
            f.close()  
            
            peaksearch_win = np.arange(temporal_feat.shape[0]-40,temporal_feat.shape[0])
            idx_tempPeak = np.argmax(np.abs(temporal_feat[peaksearch_win]))     # only check for peak in the final 25 time points.
            idx_tempPeak = idx_tempPeak + peaksearch_win[0]
            sign = np.sign(temporal_feat[idx_tempPeak])
            if sign<0:
                spatial_feat = spatial_feat*sign
                temporal_feat = temporal_feat*sign

            spatRF_sta[:,:,ctr_d,m] = spatial_feat
            tempRF_sta[:,ctr_d,m]  = temporal_feat[-temp_window:]
            tempRF_sta[:,ctr_d,m]  = tempRF_sta[:,ctr_d,m]/tempRF_sta[:,ctr_d,m].max()
            
            # Method 2: Compute LSTA from model for just one input sample
            select_img = 50 #768 #712
            spatRF, tempRF = model.featureMaps.decompose(f_grads[select_mdl][d]['grads'][u,select_img,-temp_window:,:,:])
            rf_coords,rf_fit_img,rf_params,_ = spatRF2DFit(spatRF,tempRF=0,sig_fac=sig_fac,rot=True,sta=0,tempRF_sig=False)
            mean_rfCent = np.abs(np.nanmean(rf_fit_img))
            spatRF_singImg[:,:,ctr_d,m] = spatRF/mean_rfCent
            tempRF_singImg[:,ctr_d,m] = tempRF*mean_rfCent
            tempRF_singImg[:,ctr_d,m] = tempRF_singImg[:,ctr_d,m]/tempRF_singImg[:,ctr_d,m].max()
            
        
        f_grads.close()
 
    vmin = np.min((spatRF_singImg.min(),spatRF_indiv_avg_acrossImgs.min()))
    vmax = np.max((spatRF_singImg.max(),spatRF_indiv_avg_acrossImgs.max()))
    
    tmin = np.nanmin((tempRF_sta.min(),tempRF_singImg.min()))-0.05
    tmax = np.nanmax((tempRF_sta.max(),tempRF_singImg.max()))+0.05

    cmap_name = 'gray' #'cool' # cool
        
    frametime = 8

    temp_axis = np.arange(temp_window)
    temp_axis = np.flip(temp_axis*frametime)
    n_ticks = 10
    ticks_x = np.arange(0,temp_axis.shape[0],5) #np.linspace(0,temp_axis.shape[0],n_ticks,dtype=int)-1
    ticks_x[0] = 0
    ticks_x_labels = temp_axis[ticks_x]
    
    t_frame_ds = int(t_frame*dsFac)
    x_ticksLabel_ds = -t_frame_ds*np.arange(tempRF_sta[::dsFac].shape[0])
    x_ticksLabel_ds = np.flip(x_ticksLabel_ds)
    x_ticks_ds = np.arange(tempRF_sta[::dsFac].shape[0])

    font_tick = 14
    font_title = 14
    
    txt_title = 'Train: %s\nTest: %s\n%s'%(dataset_model,d,uname)
    
    n_conds = len(mdls_toplot)*(2*len(datasets_plot))
    plots_idx = np.arange(0,n_conds*2)
    plots_idx = np.array([[0,3],[1,4],[2,5]])

    fig,axs = plt.subplots(2,len(datasets_plot)*len(mdls_toplot)+1,figsize=(30,15))
    axs = np.ravel(axs)
    fig.suptitle(txt_title,size=28)
    
    ctr_d = -1
    d = dataset_eval[0]
        
        
    for d in datasets_plot:
        ctr_d+=1

        # idx_p = len(dataset_eval)*ctr_d
        # idx_p = plots_idx[0,0,ctr_d,m]
        idx_p = plots_idx[0,0]
        axs[idx_p].set_title('Conventional STA',fontsize=font_title)
        axs[idx_p].imshow(spatRF_sta[:,:,ctr_d,m],aspect='auto',cmap=cmap_name)
        axs[idx_p].axes.xaxis.set_visible(False)
        axs[idx_p].axes.yaxis.set_visible(False)

        # idx_p = plots_idx[0,1,ctr_d,m]
        idx_p = plots_idx[0,1]
        axs[idx_p].plot(tempRF_sta[::dsFac,ctr_d,m])
        # axs[idx_p].set_xlabel('Time prior to spike (ms)',size=font_tick)
        axs[idx_p].set_xticks(x_ticks_ds)
        axs[idx_p].set_xticklabels(x_ticksLabel_ds)
        axs[idx_p].set_ylim(tmin,tmax)
        axs[idx_p].set_ylabel('R*/rod/sec',size=font_tick)
        axs[idx_p].tick_params(axis='both',labelsize=font_tick)

        for m in range(len(mdls_toplot)):
            select_mdl = mdls_toplot[m]
           
            
            # idx_p = plots_idx[1,0,ctr_d,m]
            idx_p = plots_idx[m+1,0]
            axs[idx_p].set_title('single sample',fontsize=font_title)
            axs[idx_p].imshow(spatRF_singImg[:,:,ctr_d,m],aspect='auto',cmap=cmap_name)#,vmin=-vmax,vmax=-vmin)
            axs[idx_p].axes.xaxis.set_visible(False)
            axs[idx_p].axes.yaxis.set_visible(False)
            # idx_p = plots_idx[1,1,ctr_d,m]
            
            idx_p = plots_idx[m+1,1]
            txt_subtitle = '%s | %s | FEV = %02d%%'%(select_mdl,d[5:],perf_datasets[select_mdl][d]['fev_allUnits'][select_rgc_dataset]*100)
            axs[idx_p].set_title(txt_subtitle,fontsize=font_title)
            axs[idx_p].plot(tempRF_singImg[:,ctr_d,m])
            # axs[idx_p].set_xlabel('Time prior to spike (ms)',size=font_tick)
            axs[idx_p].set_xticks(ticks_x)
            axs[idx_p].set_ylim(tmin,tmax)
            axs[idx_p].set_xticklabels(ticks_x_labels)
            axs[idx_p].tick_params(axis='both',labelsize=font_tick)
            axs[idx_p].set_ylabel('spikes/R*/rod',size=font_tick)


# fig_name = 'sta_vs_grads'
# fig.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
# fig.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)



# %% Grads vs STA Response kinetics


fname_gainFile = '/home/saad/postdoc_db/analyses/data_kiersten/monkey01/gradient_analysis/gain_analysis_ds.h5'
fname_stas =  '/home/saad/postdoc_db/analyses/data_kiersten/monkey01/db_files/datasets/monkey01_STAs_allLightLevels_8ms_Rstar.h5'

f = h5py.File(fname_gainFile,'r')
select_lightLevel = 'scot-0.3-Rstar'

uname_gainFile = list(f[select_mdl][select_lightLevel].keys()) #['on_mid_003', 'on_mid_004', 'on_mid_005', 'on_mid_006', 'on_mid_009', 'on_mid_011', 'on_mid_015', 'on_mid_016', 'on_mid_017', 'on_mid_018', 'on_mid_020']

select_mdl = 'PRFR_CNN2D_RODS'  # CNN_2D_NORM # PRFR_CNN2D_RODS
lightLevels_all = ('scot-30-Rstar','scot-3-Rstar','scot-0.3-Rstar')

temp_win = 40
dsFac = 4
temp_win_ds = int(temp_win/dsFac)
nbins = 10
    
lat_cnn = np.zeros((len(uname_gainFile),len(lightLevels_all)))
lat_pr = np.zeros((len(uname_gainFile),len(lightLevels_all)))
lat_sta = np.zeros((len(uname_gainFile),len(lightLevels_all)))

gain_grads_cnn = np.zeros((nbins,len(uname_gainFile)));gain_grads_cnn[:]=np.nan
tempRF_grads_cnn = np.zeros((temp_win,nbins,len(uname_gainFile)));tempRF_grads_cnn[:]=np.nan
tempRF_grads_cnn_avg = np.zeros((temp_win,len(lightLevels_all),len(uname_gainFile)));tempRF_grads_cnn_avg[:]=np.nan
fevs_cnn = np.zeros((len(uname_gainFile)));fevs_cnn[:]=np.nan

gain_grads_pr = np.zeros((nbins,len(uname_gainFile)));gain_grads_cnn[:]=np.nan
tempRF_grads_pr = np.zeros((temp_win,nbins,len(uname_gainFile)));tempRF_grads_cnn[:]=np.nan
tempRF_grads_pr_avg = np.zeros((temp_win,len(lightLevels_all),len(uname_gainFile)));tempRF_grads_pr_avg[:]=np.nan
fevs_pr = np.zeros((len(uname_gainFile)));fevs_pr[:]=np.nan

tempRF_sta = np.zeros((temp_win,len(lightLevels_all),len(uname_gainFile)));tempRF_sta[:]=np.nan
tempRF_sta_ds = np.zeros((temp_win_ds,len(lightLevels_all),len(uname_gainFile)));tempRF_sta_ds[:]=np.nan

d = -1
binsToTake = np.array([0,1,2,3,4,5,6,7,8,9])

select_lightLevel = 'scot-0.3-Rstar'
for select_lightLevel in lightLevels_all:
    d += 1
    uname_gainFile = list(f[select_mdl][select_lightLevel].keys()) #['on_mid_003', 'on_mid_004', 'on_mid_005', 'on_mid_006', 'on_mid_009', 'on_mid_011', 'on_mid_015', 'on_mid_016', 'on_mid_017', 'on_mid_018', 'on_mid_020']
    

    for u in range(len(uname_gainFile)):
        uname = uname_gainFile[u]
        gain_grads_cnn[:,u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['gain_grads_binned'])
        tempRF_grads_cnn[:,:,u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['tempRF_grads_binned_grand'][-temp_win:])
        rgb = np.nanmean(tempRF_grads_cnn[:,binsToTake,u],axis=1)
        tempRF_grads_cnn_avg[:,d,u] = rgb/np.nanmax(rgb,axis=0,keepdims=True)
        fevs_cnn[u] = np.array(f['CNN_2D_NORM'][select_lightLevel][uname]['fev'])
        
        gain_grads_pr[:,u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['gain_grads_binned'])
        tempRF_grads_pr[:,:,u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['tempRF_grads_binned_grand'][-temp_win:])
        rgb = np.nanmean(tempRF_grads_pr[:,binsToTake,u],axis=1)
        tempRF_grads_pr_avg[:,d,u] = rgb/np.nanmax(rgb,axis=0,keepdims=True)
        fevs_pr[u] = np.array(f['PRFR_CNN2D_RODS'][select_lightLevel][uname]['fev'])
        
        f_sta = h5py.File(fname_stas,'r')
        spatial_feat = np.array(f_sta[select_lightLevel[:-6]][uname]['spatial_feature'])
        temporal_feat = np.array(f_sta[select_lightLevel[:-6]][uname]['temporal_feature'])
        f_sta.close()  
        
        peaksearch_win = np.arange(temporal_feat.shape[0]-40,temporal_feat.shape[0])
        idx_tempPeak = np.argmax(np.abs(temporal_feat[peaksearch_win]))     # only check for peak in the final 25 time points.
        idx_tempPeak = idx_tempPeak + peaksearch_win[0]
        sign = np.sign(temporal_feat[idx_tempPeak])
        if sign<0:
            spatial_feat = spatial_feat*sign
            temporal_feat = temporal_feat*sign
        
        rgb = temporal_feat[-temp_win:]/temporal_feat[-temp_win:].max()
        tempRF_sta[:,d,u] = rgb
        tempRF_sta_ds[:,d,u] = rgb[::dsFac]
        
        
        lat_cnn[u,d] = temp_win-np.argmax(tempRF_grads_cnn_avg[:,d,u])
        lat_pr[u,d] = temp_win-np.argmax(tempRF_grads_pr_avg[:,d,u])
        lat_sta[u,d] = temp_win-np.argmax(tempRF_sta[:,d,u])

# %%
rgb = np.abs(tempRF_grads_pr_avg[:,2,:] - tempRF_grads_pr_avg[:,1,:])
a = np.sum(rgb,axis=0)
idx_max = np.argmax(a)

# %%
u = 9 # 14
d = np.array([0,1,2])
t_frame = 8

lat_stack = np.concatenate((lat_cnn[:,:,None],lat_pr[:,:,None],lat_sta[:,:,None]),axis=-1)
lab_cnn = np.reshape(np.repeat(np.array(['CNN']),lat_cnn.size),(lat_cnn.shape[0],lat_cnn.shape[1]))
lab_pr = np.reshape(np.repeat(np.array(['PR_CNN']),lat_pr.size),(lat_pr.shape[0],lat_pr.shape[1]))
lab_sta = np.reshape(np.repeat(np.array(['STA']),lat_sta.size),(lat_sta.shape[0],lat_sta.shape[1]))
lat_labels_models_stack = np.concatenate((lab_cnn[:,:,None],lab_pr[:,:,None],lab_sta[:,:,None]),axis=-1)
rgb = np.concatenate((np.array(['30']),np.array(['3']),np.array(['0.3'])))
lat_labels_ll = np.repeat(rgb[None,:],lat_cnn.shape[0],axis=0)
lat_labels_ll_stack = np.concatenate((lat_labels_ll[:,:,None],lat_labels_ll[:,:,None],lat_labels_ll[:,:,None]),axis=-1)

lat_flat = lat_stack.flatten()
lat_labels_ll_flat = lat_labels_ll_stack.flatten()
lat_labels_models_flat = lat_labels_models_stack.flatten()


lat_df = pd.DataFrame(lat_flat,columns=['latency'])
lat_df['model'] = lat_labels_models_flat
lat_df['lightLevel'] = lat_labels_ll_flat

# stats
# plt.hist(lat_pr_diff[:,0])
mean_lat_cnn = np.mean(lat_cnn,axis=0)
p_lat_cnn = np.zeros(lat_cnn.shape[-1]-1)
mean_lat_pr = np.mean(lat_pr,axis=0)
p_lat_pr = np.zeros(lat_cnn.shape[-1]-1)
mean_lat_sta = np.mean(lat_sta,axis=0)
p_lat_sta = np.zeros(lat_cnn.shape[-1]-1)

for i in range(lat_cnn.shape[-1]-1):
    try:
        _,p_lat_cnn[i] = scipy.stats.ranksums(lat_cnn[:,i],lat_cnn[:,i+1])
    except:
        pass
    _,p_lat_pr[i] = scipy.stats.ranksums(lat_pr[:,i],lat_pr[:,i+1])
    _,p_lat_sta[i] = scipy.stats.ranksums(lat_sta[:,i],lat_sta[:,i+1])

temp_axis = np.arange(temp_win)*t_frame
x_ticks = np.array([0,temp_axis.shape[0]-1]) 
x_ticksLabel = temp_axis[x_ticks]
x_ticksLabel = -np.flip(x_ticksLabel)
x_ticks_ds = np.array([0,tempRF_sta[::dsFac].shape[0]-1])
x_ticksLabel_ds = x_ticksLabel[[0,-1]]

fig,axs = plt.subplots(1,3,figsize=(20,5))
fig.suptitle(uname_gainFile[u])
axs[0].plot(tempRF_sta[:,d,u],label=np.array(lightLevels_all)[d])
axs[0].set_xticks(x_ticks)
axs[0].set_xticklabels(x_ticksLabel)
axs[0].set_title('STA')
axs[0].set_xlabel('Time from spike onset (ms)')
axs[0].set_ylabel('Normalized STA')
axs[0].legend()

axs[1].plot(tempRF_grads_cnn_avg[:,d[0],u],label=np.array(lightLevels_all)[d[0]])
axs[1].plot(tempRF_grads_cnn_avg[:,d[1],u],'--',label=np.array(lightLevels_all)[d[1]])
axs[1].plot(tempRF_grads_cnn_avg[:,d[2],u],'-.',label=np.array(lightLevels_all)[d[2]])
axs[1].set_xticks(x_ticks)
axs[1].set_xticklabels(x_ticksLabel)
axs[1].set_title('CNN Gradients')
axs[1].set_xlabel('Time from spike onset (ms)')
axs[1].set_ylabel('Normalized Gradients')
axs[1].legend()

axs[2].plot(tempRF_grads_pr_avg[:,d,u],label=np.array(lightLevels_all)[d])
axs[2].set_xticks(x_ticks)
axs[2].set_xticklabels(x_ticksLabel)
axs[2].set_title('PR_CNN Gradients')
axs[2].set_xlabel('Time from spike onset (ms)')
axs[2].set_ylabel('Normalized Gradients')
axs[2].legend()


fig2 = sns.catplot(
    data=lat_df,kind='bar',y='latency',
    x='model',hue='lightLevel',
    ci=95)

fig_name = 'latencies_examp'
fig.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
fig.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)

fig_name = 'latencies_pop'
fig2.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
fig2.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)


# %% Example RGC spike rate
import h5py
import numpy as np
import os
import re
  
from global_scripts import utils_si
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from scipy.stats import wilcoxon
import gc
 
import csv
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])

from model.load_savedModel import load
from model.data_handler import load_data, load_h5Dataset, prepare_data_cnn2d, prepare_data_cnn3d, prepare_data_convLSTM, prepare_data_pr_cnn2d, rolling_window
from model.performance import getModelParams, model_evaluate,model_evaluate_new,paramsToName, get_weightsDict, get_weightsOfLayer
from model import metrics
from model import featureMaps
from model.models import modelFileName
from pyret.filtertools import sta, decompose
import seaborn

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Input, Reshape


font_size_ticks = 24
font_size_labels = 24
font_size_legend = 18
font_size_title = 24

exp_select = 'monkey01'
BN = 1
subFold = 'ICLR2023'
path_exp = '/home/saad/data/analyses/data_kiersten/'+exp_select+'/'+subFold
mdl_params = 'U-37_T-120_C1-08-09_C2-16-07_C3-18-05_BN-'+str(BN)+'_MP-1_LR-0.001_TRSAMPS-040_TR-01'
mdl_name = 'CNN_2D_NORM'
trainingDataset = 'scot-3-30-Rstar'
val_dataset_2_all = ('scot-30-Rstar','scot-3-Rstar','scot-0.3-Rstar')
mdlFolder = os.path.join(path_exp,trainingDataset,mdl_name,mdl_params)



fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)

# col_scheme = ('#fd8d3c','#f03b20','#bd0026')
col_scheme = ('#fe9929','#d95f0e','#993404')


font_size_ticks = 24
font_size_labels = 24
font_size_legend = 18
font_size_title = 24
y_ticks_viol = np.arange(-100,140,20)
y_lim_viol = (-40,160)

N_rgcs = mdl.layers[-1].output.shape[-1]

med_fevs = 0
fev_stack = np.zeros(N_rgcs)

obs_rate_stack = np.zeros((1071,N_rgcs,len(val_dataset_2_all)))
pred_rate_stack = np.zeros((1071,N_rgcs,len(val_dataset_2_all)))

correctMedian = True
v_idx = 1
for v_idx in range(len(val_dataset_2_all)):
    
    val_dataset_2 = val_dataset_2_all[v_idx]
    
    fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
    # fname_data_train_val_test = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/monkey01/gradient_analysis/datasets/',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
    _,data_val,_,rgb,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)
    resp_orig = resp_orig['train']
    filt_width = 120
    data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
    samps_shift_2 = 0
    
    
    obs_rate = data_val.y
    obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,filt_width:,:]
    
    if correctMedian==True:
        fname_data_train_val_test_d1 = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+trainingDataset+'.h5'))
        _,_,_,data_quality,_,_,resp_med_d1 = load_h5Dataset(fname_data_train_val_test_d1)
        resp_med_d1 = np.nanmedian(resp_med_d1['train'],axis=0)
        resp_med_d2 = np.nanmedian(resp_orig,axis=0)
        resp_mulFac = resp_med_d2/resp_med_d1;
        
        obs_rate_allStimTrials = obs_rate_allStimTrials/resp_mulFac
        obs_rate = obs_rate/resp_mulFac

    
    
    obs_rate_stack[:,:,v_idx] = obs_rate

t_frame = 8
t_start = 60
t_dur = obs_rate.shape[0]
t_end = t_dur-20
win_display = (t_start,t_start+t_dur)
# %%
idx_display = 270+np.arange(0,800).astype('int')
t_axis = np.arange(0,idx_display.shape[0]*t_frame,t_frame)


col_mdl = ('r')
lim_y = (-0.2,3)
u = 11#16
for u in range(u,u+1):#range(u,u+1): #range(N_rgcs):
    idx_unitsToPred = u #14 #13

    figure_1a,axs = plt.subplots(1,1,figsize=(10,5))
    axs = np.ravel(axs)
    figure_1a.suptitle('')
    v_idx=0
    
    # l_base, = axs[v_idx].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate_stack[t_start:t_end-samps_shift_2,idx_unitsToPred,v_idx],linewidth=8,color='darkgray')
    axs[v_idx].plot(t_axis,obs_rate_stack[idx_display,idx_unitsToPred,:],linewidth=2)
    axs[v_idx].plot(t_axis,obs_rate_stack[idx_display,idx_unitsToPred,:],linewidth=2)
    
    axs[v_idx].set_ylim(lim_y)
    axs[v_idx].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[v_idx].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
    axs[v_idx].set_title('Example RGC (unit-'+str(idx_unitsToPred)+'): '+val_dataset_2,fontsize=font_size_title)
    # axs[v_idx].text(0.75,6,'Trained at photopic light level\nEvaluated at photopic light level',fontsize=font_size_legend)
    axs[v_idx].legend(loc='upper left',fontsize=font_size_legend)
    plt.setp(axs[v_idx].get_xticklabels(), fontsize=font_size_ticks)
    plt.setp(axs[v_idx].get_yticklabels(), fontsize=font_size_ticks)
    
# fig_name = 'latencies_rgc'
# figure_1a.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
# figure_1a.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)
    
