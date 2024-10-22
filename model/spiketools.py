#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:47:28 2021

@author: saad
"""
import numpy as np
import math

def MEA_spikerates(spikeTimes,sig,t_end):

    t_end = int(t_end)
    sr = 1000;
    st = 10000/sr*6.2/(60*sig)
    time_list = np.arange(-3.1,3.2,st)
    kern = np.zeros((len(time_list)))
    
    for i in range(len(time_list)):
        kern[i] = 250/sig*math.exp((1-time_list[i]**2)/2)
    
    spikes_idx = np.ceil(spikeTimes).astype('int32')
    spikes_idx = spikes_idx[spikes_idx<t_end]
    spikeCounts = np.zeros(t_end)
    spikeCounts[spikes_idx-1] = 1
    spikeRate = np.convolve(spikeCounts,kern,'same')

    return (spikeRate,spikeCounts)

def MEA_spikerates_binned(spikeTimes,sig,time_bin_edges,t_bin):
    # sig is input in units of "bins" or "frames"
    # sig = 2
    sig_ms = sig*t_bin     # t_bin has to be in ms. Tells approx. how many ms per bin / frame. The amp is based on ms.
    # st = 0.5
    sr = 60
    st = 10000/sr*6.2/(60*sig_ms)
    
    time_list = np.arange(-3.1,3.1,st)
    kern = np.zeros((len(time_list)))
    
    for i in range(len(time_list)):
        kern[i] = 250/sig_ms*math.exp((1-time_list[i]**2)/2)
    
    # plt.plot(kern)
    
    spikeCounts = np.histogram(spikeTimes,time_bin_edges)
    spikeCounts = spikeCounts[0]
    spikeRate = np.convolve(spikeCounts,kern,'same')
    # plt.plot(spikeRate)

    return (spikeRate,spikeCounts)

def MEA_spikerates_rat(spikeTimes,sig,time_bin_edges,t_bin):
    # sig is input in units of "bins" or "frames"
    # sig = 2
    sig_ms = sig*t_bin     # t_bin has to be in ms. Tells approx. how many ms per bin / frame. The amp is based on ms.
    # st = 0.5
    sr = 60
    st = 10000/sr*6.2/(60*sig_ms)
    
    time_list = np.arange(-3.1,3.1,st)
    kern = np.zeros((len(time_list)))
    
    for i in range(len(time_list)):
        kern[i] = 250/sig_ms*math.exp((1-time_list[i]**2)/2)
    
    # plt.plot(kern)
    
    spikeCounts = np.histogram(spikeTimes,time_bin_edges)
    spikeCounts= spikeCounts[0]
    spikeRate = np.convolve(spikeCounts,kern,'same')
    # plt.plot(spikeRate)

    return (spikeRate,spikeCounts)


def MEA_spikerates_binned_new(spikeTimes,sig,time_bin_edges,t_bin):
    # sig is input in units of "bins" or "frames"
    # sig_temp = 4
    sig_ms = sig*t_bin
    sr = 1000
    st = 10000/sr*6.2/(60*sig)
    
    time_list = np.arange(-3.1,3.1,st)
    kern = np.zeros((len(time_list)))
    
    for i in range(len(time_list)):
        kern[i] = 250/sig_ms*math.exp((1-time_list[i]**2)/2)
    
    # plt.plot(kern)
    
    spikeCounts = np.histogram(spikeTimes,time_bin_edges)
    spikeCounts= spikeCounts[0]
    spikeRate = np.convolve(spikeCounts,kern,'same')
    # plt.plot(spikeRate)

    return (spikeRate,spikeCounts)


# %%
# Test STAs
# idx_cell = 1
# # stim = double(stim_frames.reshape(stim_frames.shape[0],num_checkers_y,num_checkers_x,order='F'))
# flips_sta = flips
# spikes = spikeTimes_cells[idx_cell]

# nFrames = 33
# num_spikes = 5000
# i = 1000
# spikeCount = 0
# sta_mat = np.zeros((nFrames,stim_frames.shape[1]))

# for i in range(1000,num_spikes):
#     spikeCount+=1
#     last_frame = np.where(flips_sta>spikes[i])[0][0]
#     first_frame = last_frame - nFrames
    
#     sta_mat = sta_mat + stim_frames[first_frame:last_frame,:]
    
# sta_mat = sta_mat/spikeCount

# sta_spat = np.std(sta_mat,axis=0)
# idx_max = np.argmax(sta_spat)
# temporal_feature = sta_mat[:,idx_max]
# spatial_feature = sta_spat.reshape(num_checkers_y,num_checkers_x,order='F')
# plt.imshow(spatial_feature,cmap='winter');plt.show()
# plt.plot(temporal_feature);plt.show()
