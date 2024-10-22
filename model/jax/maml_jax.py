#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:26:35 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import time
import gc
import h5py
import shutil
import os
import numpy as np
import jax
from jax import numpy as jnp
import optax
from tqdm.auto import tqdm
import flax
from flax.training import train_state
from model.jax import models_jax
import copy

from model.performance import model_evaluate_new


from flax.training import orbax_utils
import orbax.checkpoint

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# from functools import partial
from jax.tree_util import Partial


import re

class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict


@jax.jit
def compute_means_across_nested_dicts(data_list):
    # Initialize dictionaries to hold stacked arrays for each variable
    nested_keys = list(data_list[0].keys())
    variable_keys = data_list[0][nested_keys[0]].keys()
    
    stacked_data = {nested_key: {var_key: [] for var_key in variable_keys} for nested_key in nested_keys}
    
    # Stack values for each variable
    for data in data_list:
        for nested_key, nested_dict in data.items():
            for var_key, value in nested_dict.items():
                stacked_data[nested_key][var_key].append(value)
    
    # Convert lists to arrays and compute the mean for each variable
    means = {nested_key: {var_key: jnp.mean(jnp.stack(values), axis=0) 
                          for var_key, values in nested_vars.items()} 
             for nested_key, nested_vars in stacked_data.items()}
    
    return means

@jax.jit
def task_loss(state,params,batch_stats,batch):
    X,y = batch
    y_pred,state = state.apply_fn({'params': params,'batch_stats': batch_stats},X,training=True,mutable=['batch_stats','intermediates'])
    batch_updates = state['batch_stats']
    intermediates = state['intermediates']
    dense_activations = intermediates['dense_activations'][0]
    loss = (y_pred - y*jax.lax.log(y_pred)).mean()

    # if training==True:
    loss = loss + weight_regularizer(params,alpha=1e-4)
    loss = loss + activity_regularizer(dense_activations,alpha=1e-4)

    return loss,(y_pred,batch_updates)

@jax.jit
def maml_fit_task(state_fit_task,batch):

    grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
    (loss,(y_pred,batch_updates)),grads = grad_fn(state_fit_task,state_fit_task.params,state_fit_task.batch_stats,batch)
    
    state_fit_task = state_fit_task.apply_gradients(grads=grads)
    state_fit_task = state_fit_task.replace(batch_stats=batch_updates)
    
    return loss,(y_pred,batch_updates),state_fit_task,grads

@jax.jit
def train_step_maml(mdl_state,batch,weights_dense):
    """
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
    
    """
    @jax.jit
    def maml_loss(state,params,batch_stats,train_x,train_y,val_x,val_y,kern,bias):
        # print('maml_loss')

        batch_train = (train_x,train_y)
        batch_val = (val_x,val_y)
        state_copy = copy.deepcopy(state)
        state_copy.params['Dense_0']['kernel'] = kern
        state_copy.params['Dense_0']['bias'] = bias
        loss_task,(y_pred,batch_updates),state_task,grads_task = maml_fit_task(state_copy,batch_train)      # 1. Fit base model to training set

        # print(grads_task['Conv_0']['bias'])
        loss_task_val,(y_pred_val,batch_updates_val) = task_loss(state_task,state_task.params,state_task.batch_stats,batch_val)     # 2. Calculate loss of resulting model but using validation set / query set
        
        kern = state_task.params['Dense_0']['kernel']
        bias = state_task.params['Dense_0']['bias']
        
        return loss_task_val,(y_pred_val,batch_updates_val),state_task,grads_task,kern,bias
    
    @jax.jit
    def loss_fn(state_lossfn,params,batch_stats,batch,weights_dense):

        train_x,train_y,val_x,val_y = batch
        kern_all,bias_all = weights_dense
        
        task_losses,(y_pred_taks,batch_updates),state_tasks,grads_task,kern_tasks,bias_tasks = jax.vmap(Partial(maml_loss,state_lossfn,params,batch_stats))\
                                                                                                (train_x,train_y,val_x,val_y,kern_all,bias_all)
        task_losses = jnp.mean(task_losses)     # MOST IMPORTANT STEP

        for key in batch_updates.keys():
            batch_updates[key]['mean'] = jnp.mean(batch_updates[key]['mean'],axis=0)
            batch_updates[key]['var'] = jnp.mean(batch_updates[key]['var'],axis=0)

        weights_dense_updated = (kern_tasks,bias_tasks)
        
        return task_losses,(y_pred_taks,batch_updates,weights_dense_updated)
    
    # Perform the update step

    params = mdl_state.params
    batch_stats = mdl_state.batch_stats

    grad_fn = jax.value_and_grad(loss_fn,argnums=1,has_aux=True)
    (loss,(y_pred,batch_updates,weights_dense_updated)),grads = grad_fn(mdl_state,params,batch_stats,batch,weights_dense)

    state_new = mdl_state
    state_new = state_new.apply_gradients(grads=grads)
    state_new = state_new.replace(batch_stats=batch_updates)
    
    # Update the new state with weights from the 0th task
    state_new.params['Dense_0']['kernel'] = weights_dense_updated[0][0]
    state_new.params['Dense_0']['bias'] = weights_dense_updated[1][0]

    
    return loss,y_pred,state_new,weights_dense_updated



def eval_step(state,data,n_batches=1e5):
    def task_loss_eval(state,params,batch_stats,batch):
        X,y = batch
        y_pred,batch_updates = state.apply_fn({'params': params,'batch_stats': batch_stats},X,training=True,mutable=['batch_stats','intermediates'])
        loss = (y_pred - y*jax.lax.log(y_pred)).mean()
        return loss,(y_pred,batch_updates)
    
    if type(data) is tuple:
        X,y = data
        loss,(y_pred,batch_updates) = task_loss_eval(state,state.params,state.batch_stats,data)
        return loss,y_pred,y
    
    else:       # if the data is in dataloader format
        y_pred = jnp.empty((0,len(state.params['Dense_0']['bias'])))
        y = jnp.empty((0,len(state.params['Dense_0']['bias'])))
        loss = []
        count_batch = 0
        for batch in data:
            if count_batch<n_batches:
                X_batch,y_batch = batch
                loss_batch,(y_pred_batch,batch_updates) = task_loss_eval(state,state.params,state.batch_stats,batch)
                loss.append(loss_batch)
                y_pred = jnp.concatenate((y_pred,y_pred_batch),axis=0)
                y = jnp.concatenate((y,y_batch),axis=0)
                count_batch+=1
            else:
                break
    return loss,y_pred,y


def dict_subset(old_dict,exclude_list):
    new_dict = {}
    keys_all = list(old_dict.keys())
    for item in keys_all:
        for key_exclude in exclude_list:
            rgb = re.search(key_exclude,item,re.IGNORECASE)
            if rgb==None:
                new_dict[item] = old_dict[item]
    return new_dict

@jax.jit
def activity_regularizer(activations,alpha=1e-4):
    l1_penalty = alpha*jnp.mean(jnp.abs(activations))
    return l1_penalty

@jax.jit     
def weight_regularizer(params,alpha=1e-4):
    regularizer_exclude_list = ['BatchNorm',]
    params_subset = dict_subset(params,regularizer_exclude_list)
    
    l2_loss=0
    for w in jax.tree_leaves(params_subset):
        l2_loss = l2_loss + alpha * (w**2).mean()
    return l2_loss

    
def save_epoch(state,config,weights_dense,fname_cp):
    if os.path.exists(fname_cp):
        shutil.rmtree(fname_cp)  # Remove any existing checkpoints from the last notebook run.
    ckpt = {'model': state, 'config': config}
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(fname_cp, ckpt, save_args=save_args)
    fname_weights_dense = os.path.join(fname_cp,'weights_dense.h5')
    with h5py.File(fname_weights_dense,'w') as f:
        f.create_dataset('weights_dense_kernel',data=np.array(weights_dense[0],dtype='float32'),compression='gzip')
        f.create_dataset('weights_dense_bias',data=np.array(weights_dense[1],dtype='float32'),compression='gzip')


    


def create_learning_rate_scheduler(lr_schedule):
    print(lr_schedule)

    if lr_schedule['name'] == 'exponential_decay':
        learning_rate_fn = optax.exponential_decay(
                            init_value=lr_schedule['lr_init'], 
                            transition_steps=lr_schedule['transition_steps'], 
                            decay_rate=lr_schedule['decay_rate'], 
                            staircase=lr_schedule['staircase'],
                            transition_begin=lr_schedule['transition_begin'],
                            )
    else:
        learning_rate_fn = optax.constant_schedule(value=lr_schedule['lr_init'])
        
    return learning_rate_fn


# %% Training func

def train_maml(mdl_state,weights_dense,config,dataloader_train,dataloader_val,nb_epochs,path_model_save,save=True,lr_schedule=None,step_start=0):
    learning_rate_fn = create_learning_rate_scheduler(lr_schedule)
    
    loss_epoch_train = []
    loss_epoch_val = []
    
    loss_batch_train = []
    loss_batch_val = []
    
    epoch=0
    for epoch in tqdm(range(0,nb_epochs)):
        _ = gc.collect()
        loss_batch_train=[]
        # t = time.time()
        ctr=0
        # batch_train = next(iter(dataloader_train))
        for batch_train in dataloader_train:
            # elap = time.time()-t
            # print(elap)
            ctr = ctr+1
            loss,y_pred,mdl_state,weights_dense = train_step_maml(mdl_state,batch_train,weights_dense)
            # print(ctr)
            loss_batch_train.append(loss)
            
        idx_master = 0
        
        # Update the new state with weights from the idx_master task
        mdl_state.params['Dense_0']['kernel'] = weights_dense[0][idx_master]
        mdl_state.params['Dense_0']['bias'] = weights_dense[1][idx_master]

        data_val = next(iter(dataloader_val))
        batch_val = (data_val[0][idx_master],data_val[1][idx_master])
        loss_batch_val,y_pred,y = eval_step(mdl_state,batch_val)
        loss_batch_train_test,y_pred_train_test,y_train_test = eval_step(mdl_state,(batch_train[0][idx_master],batch_train[1][idx_master]))
    
        
        loss_currEpoch_train = np.mean(loss_batch_train)
        loss_currEpoch_val = np.mean(loss_batch_val)
    
        loss_epoch_train.append(np.mean(loss_currEpoch_train))
        loss_epoch_val.append(np.mean(loss_currEpoch_val))
        
        current_lr = learning_rate_fn(mdl_state.step)
        
        temporal_width_eval = data_val[0][idx_master].shape[1]
        fev_val,_,predCorr_val,_ = model_evaluate_new(y,y_pred,temporal_width_eval,lag=0,obs_noise=0)
        fev_val_med,predCorr_val_med = np.median(fev_val),np.median(predCorr_val)
        fev_train,_,predCorr_train,_ = model_evaluate_new(y_train_test,y_pred_train_test,temporal_width_eval,lag=0,obs_noise=0)
        fev_train_med,predCorr_train_med = np.median(fev_train),np.median(predCorr_train)
    
        print(f"Epoch: {epoch + 1}, loss: {loss_currEpoch_train:.2f}, fev: {fev_train_med:.2f}, corr: {predCorr_train_med:.2f} ||| val loss: {loss_currEpoch_val:.2f},fev_val: {fev_val_med:.2f}, corr_val: {predCorr_val_med:.2f} ||| lr: {current_lr:.2e}")
        
        if save == True:
            fname_cp = os.path.join(path_model_save,'epoch-%03d'%epoch)
            save_epoch(mdl_state,config,weights_dense,fname_cp)

    return loss_epoch_train,loss_epoch_val,mdl_state,weights_dense


# %% Finetuning

def create_mask(params,layers_finetune):
    def unfreeze_layer(layer_name_unfreeze, sub_dict):
        return {k: jax.tree_map(lambda _: True, v) if layer_name_unfreeze in k else v for k, v in sub_dict.items()}
    
    mask = jax.tree_util.tree_map(lambda _: False, params)      # Freeze all layers
    
    layers_finetune_exact = models_jax.get_exactLayers(mask,layers_finetune)
    for layer in layers_finetune_exact:
        mask = unfreeze_layer(layer, mask)  # unfreeze selected layer
        
    return mask
    

def update_optimizer(ft_mdl_state,mask=None,lr_schedule=None):
    params = ft_mdl_state.params
    
    scheduler_fn = create_learning_rate_scheduler(lr_schedule)
    optimizer = optax.adam(learning_rate=scheduler_fn)
    if mask is not None:
        optimizer = optax.chain(optax.masked(optax.adam(learning_rate=scheduler_fn),mask))
        
    ft_optim_state = optimizer.init(params=params)
    ft_mdl_state = ft_mdl_state.replace(tx=optimizer,opt_state=ft_optim_state)

        
    return ft_mdl_state
