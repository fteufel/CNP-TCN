#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Felix Teufel March 2020
MGP-TCN module
"""
import sys
sys.path.insert(0, '../../src')
import tensorflow as tf
import numpy as np
from tcn.tcn import CausalConv1D, TemporalBlock, TemporalConvNet
from utils.util import compute_global_l2
from cnp.cnp import DeterministicModel


class CNPTCN():
    """
    CNP-TCN Model: takes TCN, CNP params and data and predicts and returns loss
    """
    def __init__(self, tcn_params, cnp_params, class_imb, L2_penalty):
        #n_channels, levels, kernel_size, dropout, self.n_classes = tcn_params 
        #unpack tcn parameters    
        n_channels = tcn_params['n_channels']
        levels = tcn_params['levels']
        kernel_size = tcn_params['kernel_size']
        dropout = tcn_params['dropout']
        self.n_classes = tcn_params['n_classes']
        self.n_data_channels = cnp_params['num_channels']
        #encoder_output_sizes = cnp_params['encoder_output_sizes']
        #decoder_output_sizes = cnp_params['decoder_output_sizes']
        encoder_output_sizes = [cnp_params['encoder_output_size'] for i in range(cnp_params['encoder_levels']) ]
        decoder_output_sizes = [cnp_params['decoder_output_size'] for i in range(cnp_params['decoder_levels']-1)]
        decoder_output_sizes.append(2)

        self.cnp = DeterministicModel(encoder_output_sizes, decoder_output_sizes,self.n_data_channels, compute_loss= True)
        self.tcn = TemporalConvNet([n_channels] * levels, kernel_size, dropout)
        
        self.class_imb = class_imb
        self.L2_penalty = L2_penalty

    def __call__(self, x_input,y_input,z_input,obs_ind, x_target, z_target, obs_ind_target, labels, training = False): #need this annoying training flag for TCN dropout
        """
        The CNP-TCN model forward pass.
        Inputs:

        Outputs:
            logits: shape batch_size x n_classes, classification logits for each observation
            cnp_log_prob: batch_size, mean log_prob for the context points for each observation
            tcn_loss: 0, overall loss of the tcn classification of the batch
        """

        #1 feed the data trough cnp
        Z, cnp_log_prob = self._get_cnp_output(x_input,y_input,z_input,obs_ind, x_target, z_target, obs_ind_target)



        logits = self._get_TCN_logits_from_Z(Z, training)

        tcn_loss = self.classification_loss(logits, labels)
        #logits = tf.Print(logits, [tf.shape(tcn_loss)], message= 'shape of the tcn loss')
        #logits = tf.Print(logits, [tf.shape(logits)], message= 'shape of logits')
        #logits = tf.Print(logits, [tf.shape(cnp_log_prob)], message= 'shape of cnp log prob')
        return logits, cnp_log_prob, tcn_loss
        

        #mu_target is in format num_target_points x 1
        #need to separate per observation, per channel and pad


    ###06/03/2020 before i left make loss happen, see if Tcn works, add labels to input data

    def compute_probs(self, logits):
        probs = tf.exp(logits[:,1] - tf.reduce_logsumexp(logits, axis = 1))
        return probs

    def classification_loss(self, preds, labels):
        loss = tf.reduce_sum(
                tf.nn.weighted_cross_entropy_with_logits(
                    logits=preds, targets=labels, pos_weight=self.class_imb)
                    ) 
        if self.L2_penalty is not None:
            loss_reg = compute_global_l2() # normalized per weight! hence, use large lambda!
            loss = loss + loss_reg * self.L2_penalty
        return loss


    def _get_cnp_output(self, x_input,y_input,z_input,obs_ind, x_target, z_target, obs_ind_target):

        mu_target, sigma_target, reconstruction_log_prob = self.cnp(x_input,y_input,z_input,obs_ind, x_target, z_target, obs_ind_target)

        #mu_target is in format num_target_points x 1
        #tcn wants batch_size x n_timepoints x n_channel tensor
        #need to separate per observation, per channel and pad
        N =  tf.shape(tf.unique(obs_ind_target)[0])[0] #get batch size
        batch_array = tf.TensorArray(dtype=tf.float32, size=N)
        init_state = (0, batch_array)

        #padding: get the max number of timesteps

        max_time = tf.reduce_max(x_target)
        def cond(i,L):
            return i<N

        def body(i,batch_array):
            
            #get the mu for observation i
            i_indices = tf.where(tf.equal(obs_ind_target,i))
            mu_i = tf.squeeze(tf.gather(mu_target, i_indices))

            # time_ind, channel_ind [i] are the coordinates of mu[i] on the grid
            time_ind = tf.squeeze(tf.gather(x_target, i_indices))
            channel_ind = tf.squeeze(tf.gather(z_target, i_indices))

            #stack it (probably not necessary, but so i can keep track if my reshaping works bc i carry channel and time indicators with the data)
            stacked = tf.stack([mu_i,time_ind])#,channel_ind])
            Z = tf.reshape(stacked, [2,-1, self.n_data_channels])[0]

            num_pad = max_time - tf.reduce_max(time_ind)
            Z = tf.pad(Z, [[num_pad,0],[0,0]])
            #stacked.reshape(3,-1,44)[0].T  # np.squeeze(_mu[indices]).reshape(-1,44) gives the same for the first dim
            #reshape
            return i+1, batch_array.write(i, Z)

        n, batch_array_filled = tf.while_loop(cond, body, init_state)

        batch_stacked = batch_array_filled.stack()

        #Z needs shape
        #batch_stacked.set_shape([100, self.n_data_channels, 100]) #batch_size, 44, padded-length

        return batch_stacked, reconstruction_log_prob


    
    def _get_TCN_logits_from_Z(self, Z, training):#, n_classes, training):
        """
        Feeds the Z values (CNP output) to the TCN 
        
        returns:
            predictions (unnormalized log probabilities) for each MC sample of each obs
        """
        tcn_logits = tf.layers.dense(
            self.tcn(Z, training=training)[:, -1, :],
            self.n_classes, activation=None, 
            kernel_initializer=tf.orthogonal_initializer(),
            name='last_linear' 
        ) 

        return tcn_logits




