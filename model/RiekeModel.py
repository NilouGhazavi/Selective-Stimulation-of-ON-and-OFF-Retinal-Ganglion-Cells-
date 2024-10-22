#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:19:23 2021

@author: saad
"""

import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp, odeint
import scipy


# RungeKutta('dcdt',params,TimeStep,g[pnt-1],c[pnt-1],method=method_rk)

def drdt(params,x,y):
    # output = (params['gamma']*x)-(params['sigma']*y)
    output = -params['sigma']*y
    return output

def dpdt(params,x,y):
    output = x - (params['phi'] * y) + params['eta'] 
    return output
    
def dcdt(params,x,y):
    output = params['cur2ca'] * (params['k'] * x **params['h'])/2 - params['beta'] * y
    return output

def dcslowdt(params,x,y):
    output = params['betaSlow'] * (y-x)
    return output

def dgdt(params,x,y):
    output = x - (params['p'] * y)
    return output


def RiekeModel(params,stim_photons,ode_solver='RungeKutta'):
    
    method_rk = 'RK5'

    def RungeKutta(func_name,params,TimeStep,x_prev,y_prev,method='RK4'):
        func_to_compute = globals()[func_name]
        
        if method=='RK4':
            k1 = TimeStep * func_to_compute(params,x_prev,y_prev)
            k2 = TimeStep * func_to_compute(params,x_prev+(0.5*TimeStep),y_prev+(0.5*k1))
            k3 = TimeStep * func_to_compute(params,x_prev+(0.5*TimeStep),y_prev+(0.5*k2))
            k4 = TimeStep * func_to_compute(params,x_prev+TimeStep,y_prev+k3)
            
            y = y_prev + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    
        elif method == 'RK5':
            a = np.array([[0.0,0.0,0.0,0.0,0.0],
                  [0.25,0.0,0.0,0.0,0.0],
                  [3.0/32.0,9.0/32.0,0.0,0.0, 0.0],
                  [1932.0/2197.0, -7200.0/2197.0,  7296.0/2197.0, 0.0, 0.0],
                  [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0],
                  [-8.0/27.0,2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0 ]])
            
            b = np.array([ 16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0 ])
            c = np.array([ 0.0, 0.25, 3.0/8.0, 12.0/13.0, 1.0, 0.5 ])
            d = np.array([ 25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0 ])
            
            k1 = TimeStep * func_to_compute(params,x_prev+(c[0]*TimeStep),y_prev)
            k2 = TimeStep * func_to_compute(params,x_prev+(c[1]*TimeStep),y_prev+(a[0,0] * k1))
            k3 = TimeStep * func_to_compute(params,x_prev+(c[2]*TimeStep),y_prev+(a[1,0] * k1)+(a[1,1] * k2))
            k4 = TimeStep * func_to_compute(params,x_prev+(c[3]*TimeStep),y_prev+(a[2,0] * k1)+(a[2,1] * k2)+(a[2,2] * k3))
            k5 = TimeStep * func_to_compute(params,x_prev+(c[4]*TimeStep),y_prev+(a[3,0] * k1)+(a[3,1] * k2)+(a[3,2] * k3)+(a[3,3] * k4))
            k6 = TimeStep * func_to_compute(params,x_prev+(c[5]*TimeStep),y_prev+(a[4,0] * k1)+(a[4,1] * k2)+(a[4,2] * k3)+(a[4,3] * k4)+(a[4,4] * k5))
            
            y = y_prev + b[0]*k1 + b[1]*k2 + b[2]*k3 + b[3]*k4 + b[4]*k5 + b[5]*k6
            
            # if any(np.isnan(k1)) or any(np.isnan(k2)) or any(np.isnan(k3)) or any(np.isnan(k4)) or any(np.isnan(k5)) or any(np.isnan(k6)):
            #     y = 'error'


        elif method == 'RK45':
            a = np.array([[0,0,0,0,0,0,0],
                  [0.20, 0, 0, 0, 0, 0, 0],
                  [3/40, 9/40, 0, 0, 0, 0, 0],
                  [44/45, 56/15,  32/9, 0, 0,0,0],
                  [19372/6561, 25360/2187, 64448/6561,212/729, 0,0,0],
                  [9017/3168,355/33,46732/5247,49/176,5103/18656,0,0],
                  [35/384,0,500/1113,125/192,2187/6784,11/84,0]])
                  #[5179/57600,0,7571/16695,393/640,92097/339200,187/2100,1/40]])

            a = np.array([[0,0,0,0,0,0,0],
                  [1, 0, 0, 0, 0, 0, 0],
                  [1/4, 3/4, 0, 0, 0, 0, 0],
                  [11/9, -14/3,  40/9, 0, 0,0,0],
                  [4843/1458, -3170/243, 8056/729,-53/162, 0,0,0],
                  [9017/3168,-355/33,46732/5247,49/176,-5103/18656,0,0],
                  [35/384,0,500/1113,125/192,-2187/6784,11/84,0]])
            
            b = np.array([5179/57600,0,7571/16695,393/640,92097/339200,187/2100,1/40])
            # b = np.array([35/384,0,500/1113,125/192,2187/6784,11/84,0])
            c = np.array([0,1/5, 3/10, 4/5, 8/9, 1, 1])
            
            k1 = TimeStep * func_to_compute(params,x_prev+(c[0]*TimeStep),y_prev)
            k2 = TimeStep * func_to_compute(params,x_prev+(c[1]*TimeStep),y_prev+(a[1,0] * k1))
            k3 = TimeStep * func_to_compute(params,x_prev+(c[2]*TimeStep),y_prev+(a[2,0] * k1)+(a[2,1] * k2))
            k4 = TimeStep * func_to_compute(params,x_prev+(c[3]*TimeStep),y_prev+(a[3,0] * k1)+(a[3,1] * k2)+(a[3,2] * k3))
            k5 = TimeStep * func_to_compute(params,x_prev+(c[4]*TimeStep),y_prev+(a[4,0] * k1)+(a[4,1] * k2)+(a[4,2] * k3)+(a[4,3] * k4))
            k6 = TimeStep * func_to_compute(params,x_prev+(c[5]*TimeStep),y_prev+(a[5,0] * k1)+(a[5,1] * k2)+(a[5,2] * k3)+(a[5,3] * k4)+(a[5,4] * k5))
            k7 = TimeStep * func_to_compute(params,x_prev+(c[6]*TimeStep),y_prev+(a[6,0] * k1)+(a[6,1] * k2)+(a[6,2] * k3)+(a[6,3] * k4)+(a[6,4] * k5)+(a[6,5] * k6))
            
            y = y_prev + b[0]*k1 + b[1]*k2 + b[2]*k3 + b[3]*k4 + b[4]*k5 + b[5]*k6 + b[6]*k7
            
            # if any(np.isnan(k1)) or any(np.isnan(k2)) or any(np.isnan(k3)) or any(np.isnan(k4)) or any(np.isnan(k5)) or any(np.isnan(k6)) or any(np.isnan(k7)):
            #     y = 'error'


            
        return y
    
        
    NumPts = params['tme'].shape[0]
    if stim_photons.ndim > 1:
        NumPixels = stim_photons.shape[-1]
    else:
        NumPixels = 1
    TimeStep = params['tme'][1] - params['tme'][0]
    
    cdark = params['cdark']
    cgmphill=params['h']
    cgmp2cur = params['k']
    
    params['gdark'] = (2 * params['darkCurrent'] / cgmp2cur) **(1/cgmphill)
    
    cur2ca = params['beta'] * cdark / params['darkCurrent'];                # get q using steady state
    smax = params['eta']/params['phi'] * params['gdark'] * (1 + (cdark / params['hillaffinity']) **params['hillcoef'])		# get smax using steady state
    
    params['cur2ca'] = cur2ca
    
    g     = np.zeros((NumPts,NumPixels)) # free cgmp
    s     = np.zeros((NumPts,NumPixels)) # cgmp synthesis rate
    c     = np.zeros((NumPts,NumPixels)) # free calcium concentration
    p     = np.zeros((NumPts,NumPixels)) # pde activity
    r     = np.zeros((NumPts,NumPixels)) # rhodopsin activity
    cslow = np.zeros((NumPts,NumPixels))

    # initial conditions
    g[0] = params['gdark']
    s[0] = params['gdark'] * params['eta']/params['phi']
    c[0] = cdark
    r[0] = stim_photons[0] * params['gamma'] / params['sigma']
    p[0] = (params['eta'] + r[0])/params['phi']
    cslow[0] = cdark
     
    
        
    # solve difference equations
    if ode_solver=='hybrid':

        TimeStep_new = 1e-3#TimeStep
        resampFac = int(TimeStep/TimeStep_new)
        x = stim_photons.copy() / TimeStep
        x = x * TimeStep_new
        x = np.repeat(x,resampFac,axis=0)
        tm = np.arange(0,x.shape[0])*TimeStep_new
        
        r_kern = np.exp(-params['sigma']*tm)
        out_r = np.apply_along_axis(lambda m: scipy.signal.convolve(m,r_kern), axis=0, arr=x)*params['gamma']
        out_r = out_r[:x.shape[0],:] + ((x[0,:] * params['gamma'] / params['sigma'])*r_kern[:,None])
        out_r = np.concatenate((np.atleast_1d(r[0])[None,:],out_r[:-1,:]))
        # out_r = out_r[::resampFac]
        # r = out_r
        
        p_kern = np.exp(-params['phi']*tm)
        out_p = np.apply_along_axis(lambda m: scipy.signal.convolve(m,p_kern), axis=0, arr=out_r)*TimeStep_new
        out_p = out_p[:x.shape[0],:]
        out_p = out_p + (params['eta']/params['phi'])+(((x[0,:] * params['gamma']) / (params['sigma']*params['phi']))*p_kern[:,None])
        # p = out_p
        
        r = out_r[::resampFac]
        p = out_p[::resampFac]


        
        for pnt in range(1,NumPts):           
            c[pnt] = RungeKutta('dcdt',params,TimeStep,g[pnt-1],c[pnt-1],method=method_rk)
            s[pnt] = smax / (1 + (c[pnt] / params['hillaffinity']) **params['hillcoef'])
            params['p'] = p[pnt-1]
            rgb_g = RungeKutta('dgdt',params,TimeStep,s[pnt-1],g[pnt-1],method=method_rk)
            idx_nan = np.isnan(rgb_g)
            rgb_g[idx_nan] = 0 #-(cgmp2cur * params['gdark'] **cgmphill)/2
            g[pnt] = rgb_g


    elif ode_solver=='RungeKutta':
        for pnt in range(1,NumPts):           
            with np.errstate(all='ignore'):
                r[pnt] = RungeKutta('drdt',params,TimeStep,stim_photons[pnt-1],r[pnt-1],method=method_rk)
                r[pnt] = r[pnt] + params['gamma'] * stim_photons[pnt-1]
                p[pnt] = RungeKutta('dpdt',params,TimeStep,r[pnt-1],p[pnt-1],method=method_rk)
                c[pnt] = RungeKutta('dcdt',params,TimeStep,g[pnt-1],c[pnt-1],method=method_rk)
                s[pnt] = smax / (1 + (c[pnt] / params['hillaffinity']) **params['hillcoef'])
                params['p'] = p[pnt-1]
                rgb_g = RungeKutta('dgdt',params,TimeStep,s[pnt-1],g[pnt-1],method=method_rk)
                idx_nan = np.isnan(rgb_g)
                rgb_g[idx_nan] = 0 #-(cgmp2cur * params['gdark'] **cgmphill)/2
                g[pnt] = rgb_g

            
    elif ode_solver=='Euler':
        for pnt in range(1,NumPts):     
            r[pnt] = r[pnt-1] + TimeStep * (-params['sigma'] * r[pnt-1])
            r[pnt] = r[pnt] + params['gamma'] * stim_photons[pnt-1]
            p[pnt] = p[pnt-1] + TimeStep * (r[pnt-1] + params['eta'] - params['phi'] * p[pnt-1])
            # c[pnt] = c[pnt-1] + TimeStep * (cur2ca * (cgmp2cur * g[pnt-1] **cgmphill)/2 - params['beta'] * c[pnt-1])
            c[pnt] = c[pnt-1] + TimeStep * (cur2ca * cgmp2cur * g[pnt-1]**cgmphill /(1+(cslow[pnt-1]/cdark)) - params['beta'] * c[pnt-1])
            cslow[pnt] = cslow[pnt-1] - TimeStep * (params['betaSlow'] * (cslow[pnt-1]-c[pnt-1]))
            s[pnt] = smax / (1 + (c[pnt] / params['hillaffinity']) **params['hillcoef'])
            
            rgb_g = g[pnt-1] + TimeStep * (s[pnt-1] - p[pnt-1] * g[pnt-1])
            # rgb_g = g[pnt-1] + TimeStep * (s[pnt-1] - 1 * g[pnt-1])
            idx_nan = np.isnan(rgb_g)
            rgb_g[idx_nan] = 0 #-(cgmp2cur * params['gdark'] **cgmphill)/2
            g[pnt] = rgb_g


    with np.errstate(over='ignore'):
        response = -(cgmp2cur * g **cgmphill)/2
    response = np.squeeze(response)
    params['p'] = p
    params['g'] = g
    params['c'] = c
    params['cslow'] = cslow
    params['s'] = s
    params['r'] = r
        
        
    return params,response


@tf.function
def RiekeModel_tf(params,stim_photons_tf):
      
    sigma = params['sigma']
    phi = params['phi']
    eta = params['eta']
    cgmp2cur = params['k']
    cgmphill = params['h']
    cdark = params['cdark']
    beta = params['beta']
    betaSlow = params['betaSlow']
    hillcoef = params['hillcoef']
    hillaffinity = params['hillaffinity']
    gamma = params['gamma']
    timeStep = params['timeStep']
    darkCurrent = params['darkCurrent']
    gdark = params['gdark']
    TimeStep = params['timeStep']

    gdark = (2 * darkCurrent / cgmp2cur) **(1/cgmphill)
    
    cur2ca = beta * cdark / darkCurrent                # get q using steady state
    smax = eta/phi * gdark * (1 + (cdark / hillaffinity) **hillcoef)		# get smax using steady state
    
    if tf.rank(stim_photons_tf) > 1:
        NumPixels = stim_photons_tf.shape[-1]
    else:
        NumPixels = 1

    tme = tf.range(0,stim_photons_tf.shape[0],dtype='float32')*timeStep
    NumPts = tme.shape[0]
    
# initial conditions
    g_prev = tf.cast(gdark,dtype=tf.float32)
    s_prev = tf.cast(gdark * eta/phi,dtype=tf.float32)
    c_prev = tf.cast(cdark,dtype=tf.float32)
    cslow_prev = tf.cast(cdark,dtype=tf.float32)

    g_prev = tf.repeat(g_prev,NumPixels)   
    s_prev = tf.repeat(s_prev,NumPixels)    
    c_prev = tf.repeat(c_prev,NumPixels)
    cslow_prev = tf.repeat(cslow_prev,NumPixels)

    r_prev = stim_photons_tf[0] * gamma / sigma
    p_prev = (eta + r_prev)/phi


    g = tf.TensorArray(tf.float32,size=NumPts)
    g = g.write(0,g_prev)
    
    # solve difference equations
    for pnt in tf.range(1,NumPts):
        r_curr = r_prev + TimeStep * (-1 * sigma * r_prev)
        r_curr = r_curr + gamma * stim_photons_tf[pnt-1]
        p_curr = p_prev + TimeStep * (r_prev + eta - phi * p_prev)
        c_curr = c_prev + TimeStep * (cur2ca * (cgmp2cur * g_prev **cgmphill)/2 - beta * c_prev)
        cslow_curr = cslow_prev - TimeStep * (betaSlow * (cslow_prev-c_prev))
        s_curr = smax / (1 + (c_curr / hillaffinity) **hillcoef)
        g_curr = g_prev + TimeStep * (s_prev - p_prev * g_prev)

        g = g.write(pnt,g_curr)
        
        # update prev values to current
        g_prev = g_curr
        s_prev = s_curr
        c_prev = c_curr
        p_prev = p_curr
        r_prev = r_curr
        cslow_prev = cslow_curr
    
    g = g.stack()
    response = -(cgmp2cur * g **cgmphill)/2
    response = tf.squeeze(response)
    
    return params, response# %%


def RiekeModel_old(params,stim_photons):
    NumPts = params['tme'].shape[0]
    if stim_photons.ndim > 1:
        NumPixels = stim_photons.shape[-1]
    else:
        NumPixels = 1
    TimeStep = params['tme'][1] - params['tme'][0]
    
    if params['biophysFlag']==1:
        
        cdark = params['cdark']
        cgmphill=params['h']
        cgmp2cur = params['k']
        
        params['gdark'] = (2 * params['darkCurrent'] / cgmp2cur) **(1/cgmphill)
        
        cur2ca = params['beta'] * cdark / params['darkCurrent'];                # get q using steady state
        smax = params['eta']/params['phi'] * params['gdark'] * (1 + (cdark / params['hillaffinity']) **params['hillcoef']);		# get smax using steady state
        
        g     = np.zeros((NumPts,NumPixels)) # free cgmp
        s     = np.zeros((NumPts,NumPixels)) # cgmp synthesis rate
        c     = np.zeros((NumPts,NumPixels)) # free calcium concentration
        p     = np.zeros((NumPts,NumPixels)) # pde activity
        r     = np.zeros((NumPts,NumPixels)) # rhodopsin activity
        cslow = np.zeros((NumPts,NumPixels))
    
        # initial conditions
        g[0] = params['gdark'];
        s[0] = params['gdark'] * params['eta']/params['phi'];		
        c[0] = cdark;
        r[0] = stim_photons[0] * params['gamma'] / params['sigma']
        p[0] = (params['eta'] + r[0])/params['phi']
        cslow[0] = cdark
    
        # solve difference equations
        for pnt in range(1,NumPts):
            r[pnt] = r[pnt-1] + TimeStep * (-params['sigma'] * r[pnt-1])
        #     Adding Stim
            r[pnt] = r[pnt] + params['gamma'] * stim_photons[pnt-1]
            p[pnt] = p[pnt-1] + TimeStep * (r[pnt-1] + params['eta'] - params['phi'] * p[pnt-1])
            c[pnt] = c[pnt-1] + TimeStep * (cur2ca * (cgmp2cur * g[pnt-1] **cgmphill)/2 - params['beta'] * c[pnt-1])
            # c[pnt] = c[pnt-1] + TimeStep * (cur2ca * cgmp2cur * g[pnt-1] **cgmphill /(1+(cslow[pnt-1]/cdark)) - params['beta'] * c[pnt-1]);
            cslow[pnt] = cslow[pnt-1] - TimeStep * (params['betaSlow'] * (cslow[pnt-1]-c[pnt-1]))
            s[pnt] = smax / (1 + (c[pnt] / params['hillaffinity']) **params['hillcoef'])
            g[pnt] = g[pnt-1] + TimeStep * (s[pnt-1] - p[pnt-1] * g[pnt-1])
        
        # determine current change
        # ios = cgmp2cur * g. **cgmphill * 2 ./ (2 + cslow ./ cdark);
        # params['response'] = -cgmp2cur * g. **cgmphill * 1 ./ (1 + (cslow ./ cdark));
        response = -(cgmp2cur * g **cgmphill)/2
        response = np.squeeze(response)
        params['p'] = p
        params['g'] = g
        params['c'] = c
        params['cslow'] = cslow
        
    else:   # linear
        filt = params['ScFact'] * (((params['tme']/params['TauR'])**3)/(1+((params['tme']/params['TauR'])**3))) * np.exp(-((params['tme']/params['TauD']))) * np.cos(((2*np.pi*params['tme'])/params['TauP'])+(2*np.pi*params['Phi']/360));
        # filt = abs(params['ScFact']) * (1 - np.exp(-params['tme'] / abs(params['TauR'])))**abs(params['pow']) * np.exp(-params['tme'] / abs(params['TauR']));
        params['response'] = np.real(np.fft.ifft(np.fft.fft(params['stm']) * np.fft.fft(filt))); # - params['darkCurrent'];
        params['response'] = params['response'] - np.mean(params['response']);

        
    return params,response

