#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some useful functions.

Author: Michael Moor 2018 

"""
import tensorflow as tf
import numpy as np
         

def num_grid_times_to_grid_times(grid_time_nums):
    """
    takes vector of num_grid_times with size batch, returns array n_batch x max_grid_time, padded with 0s.
    """
    max_time = grid_time_nums.max()
    grid_times = np.zeros((grid_time_nums.shape[0],max_time))
    for i, n in enumerate(grid_time_nums):
        times = np.arange(n)
        grid_times[i,:n] = times
        
    return grid_times

#utility function to count the number of trainable parameters:
def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        name = variable.name
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        print(name, [dim for dim in shape], variable_parameters)
        total_parameters += variable_parameters
    print('Number of trainable parameters = {}'.format(total_parameters))
    return total_parameters

def get_tcn_window(kernel_size, n_levels):
    window = 1
    for i in range(n_levels):
        window += 2**i * (kernel_size-1)
    return window

def compute_global_l2():
    variables = tf.trainable_variables()
    weights = [v for v in variables if 'kernel' in v.name ]
    L2 = tf.add_n([ tf.nn.l2_loss(w) for w in weights ])
    #print([w.name for w in weights])

    '''weight_names = [w.name for w in weights]
    values = sess.run(weight_names)
    weight_dims = [w.shape for w in values]'''
    
    weight_dims = [w.get_shape() for w in weights]

    n_weights = 0

    for weight_dim in weight_dims:
        n_weights_per_kernel = 1.0
        for dim in weight_dim:
            n_weights_per_kernel *= dim.value #dim 
        n_weights += n_weights_per_kernel

    print('N_weights:', n_weights)
    print('Weight Dims:', weight_dims)

    L2_per_weight = L2/n_weights
    #print(L2_per_weight.eval(session=sess))

    return L2_per_weight

def pad_rawdata(T,Y,ind_kf,ind_kt,X,meds_on_grid,covs):
    """ 
    Helper func. Pad raw data so it's in a padded array to be fed into the graph,
    since we can't pass in arrays of arrays directly.
    
    Inputs:
        arrays of data elements:
            T: array of arrays, with raw observation times
            Y,ind_kf,ind_kt: array of arrays;
                observed lab/vitals,
                indices into Y (same size)
            X: grid points
            meds_on_grid: list of arrays, each is grid_size x num_meds
            covs: matrix of baseline covariates for each patient. 
                to be tiled at each time and combined w meds
    Returns:
        Padded 2d np arrays of data, now of dim batchsize x batch_maxlen
    """
    N = np.shape(T)[0] #num in batch
    num_meds = np.shape(meds_on_grid[0])[1]
    num_covs = np.shape(covs)[1]
    
    T_lens = np.array([len(t) for t in T])
    T_maxlen = np.max(T_lens)
    T_pad = np.zeros((N,T_maxlen))
    
    Y_lens = np.array([len(y) for y in Y])
    Y_maxlen = np.max(Y_lens) 
    Y_pad = np.zeros((N,Y_maxlen))
    ind_kf_pad = np.zeros((N,Y_maxlen))
    ind_kt_pad = np.zeros((N,Y_maxlen))
    
    grid_lens = np.array([np.shape(m)[0] for m in meds_on_grid])
    grid_maxlen = np.max(grid_lens)
    meds_cov_pad = np.zeros((N,grid_maxlen,num_meds+num_covs))
    X_pad = np.zeros((N,grid_maxlen))
    
    for i in range(N):
        T_pad[i,:T_lens[i]] = T[i]
        Y_pad[i,:Y_lens[i]] = Y[i]
        ind_kf_pad[i,:Y_lens[i]] = ind_kf[i]
        ind_kt_pad[i,:Y_lens[i]] = ind_kt[i]
        X_pad[i,:grid_lens[i]] = X[i]
        meds_cov_pad[i,:grid_lens[i],:num_meds] = meds_on_grid[i]
        meds_cov_pad[i,:grid_lens[i],num_meds:] = np.tile(covs[i],(grid_lens[i],1))
                    
    return T_pad,Y_pad,ind_kf_pad,ind_kt_pad,X_pad,meds_cov_pad

def pad_rawdata_nomed(T,Y,ind_kf,ind_kt,X,covs,num_tcn_grid_times,min_pad_length): ## with med: meds_on_grid,covs -- now with: num_tcn_grid_times
    """ 
    Helper func. Pad raw data so it's in a padded array to be fed into the graph,
    since we can't pass in arrays of arrays directly.
    
    Inputs:
        arrays of data elements:
            T: array of arrays, with raw observation times
            Y,ind_kf,ind_kt: array of arrays;
                observed lab/vitals,
                indices into Y (same size)
            X: grid points
            meds_on_grid: list of arrays, each is grid_size x num_meds
            covs: matrix of baseline covariates for each patient. 
                to be tiled at each time and combined w meds
    Returns:
        Padded 2d np arrays of data, now of dim batchsize x batch_maxlen
    """
    N = np.shape(T)[0] #num in batch
    ##num_meds = np.shape(meds_on_grid[0])[1]
    num_covs = np.shape(covs)[1]
    
    T_lens = np.array([len(t) for t in T])
    T_maxlen = np.max(T_lens)
    T_maxlen = np.max([T_maxlen, min_pad_length]) #set padded time series length of batch at least to min_pad_length
    T_pad = np.zeros((N,T_maxlen))
    
    Y_lens = np.array([len(y) for y in Y])
    Y_maxlen = np.max(Y_lens) 
    Y_pad = np.zeros((N,Y_maxlen))
    ind_kf_pad = np.zeros((N,Y_maxlen))
    ind_kt_pad = np.zeros((N,Y_maxlen))
    
    ##grid_lens = np.array([np.shape(m)[0] for m in meds_on_grid])
    grid_lens = num_tcn_grid_times
    grid_maxlen = np.max(grid_lens)
    meds_cov_pad = np.zeros((N,grid_maxlen,num_covs)) ## num_meds+num_covs 
    X_pad = np.zeros((N,grid_maxlen))
    
    for i in range(N):
        T_pad[i,:T_lens[i]] = T[i]
        Y_pad[i,:Y_lens[i]] = Y[i]
        ind_kf_pad[i,:Y_lens[i]] = ind_kf[i]
        ind_kt_pad[i,:Y_lens[i]] = ind_kt[i]
        X_pad[i,:grid_lens[i]] = X[i]
        ##meds_cov_pad[i,:grid_lens[i],:num_meds] = meds_on_grid[i]
        meds_cov_pad[i,:grid_lens[i],:num_covs] = np.tile(covs[i],(grid_lens[i],1)) ## meds_cov_pad[i,:grid_lens[i],num_meds:]
                    
    return T_pad,Y_pad,ind_kf_pad,ind_kt_pad,X_pad,meds_cov_pad

def pad_rawdata_nomed_nocovs(T,Y,ind_kf,ind_kt,X,num_tcn_grid_times, min_pad_length): ## with med: meds_on_grid,covs -- now with: num_tcn_grid_times
    """ 
    Helper func. Pad raw data so it's in a padded array to be fed into the graph,
    since we can't pass in arrays of arrays directly.
    
    Inputs:
        arrays of data elements:
            T: array of arrays, with raw observation times
            Y,ind_kf,ind_kt: array of arrays;
                observed lab/vitals,
                indices into Y (same size)
            X: grid points
            meds_on_grid: list of arrays, each is grid_size x num_meds
            
    Returns:
        Padded 2d np arrays of data, now of dim batchsize x batch_maxlen
    """
    N = np.shape(T)[0] #num in batch
    ##num_meds = np.shape(meds_on_grid[0])[1]
    ##num_covs = np.shape(covs)[1]
    
    T_lens = np.array([len(t) for t in T])
    T_maxlen = np.max(T_lens)
    T_maxlen = np.max([T_maxlen, min_pad_length]) #set padded time series length of batch at least to min_pad_length
    T_pad = np.zeros((N,T_maxlen))
    
    Y_lens = np.array([len(y) for y in Y])
    Y_maxlen = np.max(Y_lens) 
    Y_pad = np.zeros((N,Y_maxlen))
    ind_kf_pad = np.zeros((N,Y_maxlen))
    ind_kt_pad = np.zeros((N,Y_maxlen))
    
    ##grid_lens = np.array([np.shape(m)[0] for m in meds_on_grid])
    grid_lens = num_tcn_grid_times
    grid_maxlen = np.max(grid_lens)
    #meds_cov_pad = np.zeros((N,grid_maxlen,num_covs)) ## num_meds+num_covs 
    X_pad = np.zeros((N,grid_maxlen))
    
    for i in range(N):
        T_pad[i,:T_lens[i]] = T[i]
        Y_pad[i,:Y_lens[i]] = Y[i]
        ind_kf_pad[i,:Y_lens[i]] = ind_kf[i]
        ind_kt_pad[i,:Y_lens[i]] = ind_kt[i]
        X_pad[i,:grid_lens[i]] = X[i]
        ##meds_cov_pad[i,:grid_lens[i],:num_meds] = meds_on_grid[i]
        ##meds_cov_pad[i,:grid_lens[i],:num_covs] = np.tile(covs[i],(grid_lens[i],1)) ## meds_cov_pad[i,:grid_lens[i],num_meds:]
    #result = {
    #    'T': T_pad,
    #    'Y': Y_pad,
    #    'X': X_pad,
    #    'ind_lvs': ind_kf_pad,
    #    'ind_kt': ind_kt_pad 
    #}                    
    return T_pad,Y_pad,ind_kf_pad,ind_kt_pad,X_pad ##,meds_cov_pad
    #return result


