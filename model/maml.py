#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:49:31 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np
from collections import namedtuple
from tqdm import tqdm, trange
from tensorflow.keras.optimizers.legacy import Adam, SGD


def chunker_maml(data,batch_size=10,k=5,mode='default'):
    if isinstance(data.X,list):
        nsamps = len(data.X)
        if nsamps%2!=0:     # we need even num of samples
            dict_temp = dict(X=data.X[1:],y=data.y[1:])
            data = namedtuple('Exptdata',dict_temp)
            data=data(**dict_temp)
        
        nsamps = len(data.X)
        nsamps_half = int(nsamps/2)
        X_support=[]
        y_support=[]
        X_query=[]
        y_query = []
        ctr=0
        idx_support = np.arange(nsamps_half)
        X_s=[]
        y_s=[]
        X_q=[]
        y_q=[]
        for i in range(nsamps_half):
            if ctr<k:
                X_s.append(data.X[i])
                y_s.append(data.y[i])
                X_q.append(data.X[nsamps_half+i])
                y_q.append(data.y[nsamps_half+i])

                ctr=ctr+1
            else:
                ctr=0
                X_support.append(X_s)
                y_support.append(y_s)
                X_s = []
                y_s = []
                X_query.append(X_q)
                y_query.append(y_q)
                X_q = []
                y_q = []
            
        counter = 0
        nsamps_tasks = len(X_support)
        while True:
            counter = (counter + 1) % nsamps_tasks

            cbatch=0
            for cbatch in range(0, nsamps_tasks, batch_size):
                yield (np.asarray(X_support[cbatch:(cbatch + batch_size)]), np.asarray(y_support[cbatch:(cbatch + batch_size)]),
                       np.asarray(X_query[cbatch:(cbatch + batch_size)]), np.asarray(y_query[cbatch:(cbatch + batch_size)]))

    else:
        if mode=='predict': # in predict mode no need to do y
            X = data
            counter = 0
            while True:
                counter = (counter + 1) % X.shape[0]
                for cbatch in range(0, X.shape[0], batch_size):
                    yield (X[cbatch:(cbatch + batch_size)])
        
        else:
            X = data.X
            y = data.y
                
            counter = 0
            while True:
                counter = (counter + 1) % X.shape[0]
                for cbatch in range(0, X.shape[0], batch_size):
                    yield (X[cbatch:(cbatch + batch_size)], y[cbatch:(cbatch + batch_size)])


# inner_optimizer = Adam(lr)
# outer_optimizer = Adam(lr)


def train_on_batch(mdl,data_batch,inner_optimizer,loss_fun,inner_step=1,outer_optimizer=None,MODE='Meta'):
    """

    """

    batch_acc = []
    task_weights = []
    loss_support = []
    
    # Save the initial weights with meta_weights and set them as the weights for the inner step model.
    meta_weights = mdl.get_weights()
    # print(meta_weights[3][0])
    X_meta_support,y_meta_support,X_meta_query,y_meta_query = data_batch
    # print(y_meta_support[0][0][5])
    # For each task, it is necessary to load the most original weights for updating.
    t=0
    for t in range(len(X_meta_support)):
        mdl.set_weights(meta_weights)

        X_support = X_meta_support[t]
        y_support = y_meta_support[t]

        with tf.GradientTape() as tape:
            y_pred = mdl(X_support, training=True)
            loss = loss_fun(y_support,y_pred)
            loss = tf.reduce_mean(loss)
            # print(loss)
            # acc = metrics.cc(y_support,y_pred)
            loss_support.append(loss)
        grads = tape.gradient(loss, mdl.trainable_variables)
        inner_optimizer.apply_gradients(zip(grads,mdl.trainable_variables))
    
    task_weights.append(mdl.get_weights())

    del tape  
    tf.keras.backend.clear_session()
    
    sum_batch_loss = 0
    batch_loss = []

    if MODE=='Meta':

        # print('test')
        t=0
        for t in range(len(X_meta_query)):
            X_query = X_meta_query[t]
            y_query = y_meta_query[t]
    
            # Load the weights for each task and perform forward propagation.
            mdl.set_weights(task_weights[t])
            with tf.GradientTape(persistent=False) as tape:
                y_pred = mdl(X_query, training=False)
                loss_query = loss_fun(y_query,y_pred)
                loss_query = tf.reduce_mean(loss_query)
                batch_loss.append(loss_query)
            # sum_batch_loss = (sum_batch_loss+loss_query)
                sum_batch_loss = tf.reduce_mean(batch_loss)
    
            # sum_batch_loss = tf.reduce_mean(batch_loss)
       
        mdl.set_weights(meta_weights)
        grads = tape.gradient(sum_batch_loss, mdl.trainable_variables)
        outer_optimizer.apply_gradients(zip(grads, mdl.trainable_variables))
            
        del grads, tape
    
    mdl_weights = mdl.get_weights()
            
    return mdl,sum_batch_loss,mdl_weights


# %% Loop over all batches
model_func = getattr(model.models_primate,mdl_name.lower())

mdl = model_func(x, n_cells, **dict_params)      
# mdl.compile(loss='poisson', optimizer=optimizer, metrics=[metrics.cc, metrics.rmse, metrics.fev],experimental_run_tf_function=False)
inner_optimizer = Adam(0.001)
outer_optimizer = Adam(0.0001)
loss_fun = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)


task_size = 2
k = 100
steps_per_epoch =  int(np.ceil(dset_details['n_train']/task_size/k/2))
gen_train = chunker_maml(data_train,task_size,k=k)
gen_val = chunker(data_val,len(data_val.X))

train_loss_batch = []
val_loss_epoch = []
nb_epochs = 100
i=0
for epoch in tqdm(range(nb_epochs)):
    for i in range(steps_per_epoch):
    # for data_batch in tqdm(gen_train):
        data_batch = next(gen_train)
        # mdl.set_weights(mdl_weights)
        mdl,sum_batch_loss,mdl_weights = train_on_batch(mdl,data_batch,inner_optimizer,loss_fun,outer_optimizer=outer_optimizer,MODE='Meta')
        train_loss_batch.append(sum_batch_loss)
    data_batch_val = next(gen_val)
    y_pred = mdl(data_batch_val[0])
    val_loss = loss_fun(data_batch_val[1],y_pred)
    val_loss = tf.reduce_mean(val_loss)
    val_loss_epoch.append(val_loss)
    print(val_loss)


weights_dict = get_weightsDict(mdl)

# %%

def train_on_batch_orig(self, train_data, inner_optimizer, inner_step, outer_optimizer=None):
        """
    MAML training process for one batch:
    :param train_data: Training data organized by task
    :param inner_optimizer: Optimizer corresponding to the support set
    :param inner_step: Number of inner update steps
    :param outer_optimizer: Optimizer corresponding to the query set; if the object does not exist, gradients are not updated
    :return: Batch query loss
        """
        batch_acc = []
        batch_loss = []
        task_weights = []

        # # Save the initial weights with meta_weights and set them as the weights for the inner step model.

        meta_weights = self.meta_model.get_weights()

        meta_support_image, meta_support_label, meta_query_image, meta_query_label = next(train_data)
        for support_image, support_label in zip(meta_support_image, meta_support_label):

            # For each task, it is necessary to load the most original weights for updating.
            self.meta_model.set_weights(meta_weights)
            for _ in range(inner_step):
                with tf.GradientTape() as tape:
                    logits = self.meta_model(support_image, training=True)
                    loss = losses.sparse_categorical_crossentropy(support_label, logits)
                    loss = tf.reduce_mean(loss)

                    acc = tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int32) == support_label, tf.float32)
                    acc = tf.reduce_mean(acc)

                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

            # After each inner loop update, the weights need to be saved to ensure that the same task is used for training in the outer loop.
            task_weights.append(self.meta_model.get_weights())

        with tf.GradientTape() as tape:
            for i, (query_image, query_label) in enumerate(zip(meta_query_image, meta_query_label)):

                # Load the weights for each task and perform forward propagation.
                self.meta_model.set_weights(task_weights[i])

                logits = self.meta_model(query_image, training=True)
                loss = losses.sparse_categorical_crossentropy(query_label, logits)
                loss = tf.reduce_mean(loss)
                batch_loss.append(loss)

                acc = tf.cast(tf.argmax(logits, axis=-1) == query_label, tf.float32)
                acc = tf.reduce_mean(acc)
                batch_acc.append(acc)

            mean_acc = tf.reduce_mean(batch_acc)
            mean_loss = tf.reduce_mean(batch_loss)

        # Regardless of whether there is an update, it is necessary to load the initial weights to prevent changes to the original weights during the validation phase.

        self.meta_model.set_weights(meta_weights)
        if outer_optimizer:
            grads = tape.gradient(mean_loss, self.meta_model.trainable_variables)
            outer_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        return mean_loss, mean_acc