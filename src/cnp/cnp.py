#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Felix Teufel March 2020

Conditional Neural Process
codebase from https://colab.research.google.com/github/deepmind/neural-processes/blob/master/conditional_neural_process.ipynb
"""
import sys
sys.path.insert(0, '../../src')
import tensorflow as tf
import numpy as np


class DeterministicEncoder(object):
    """The Encoder."""

    def __init__(self, output_sizes, n_channels):
        """CNP encoder.

        Args:
          output_sizes: An iterable containing the output sizes of the encoding MLP.
          n_channels: Number of context channels (for one-hot)
        """
        self._output_sizes = output_sizes
        self.n_channels = n_channels

    def __call__(self, context_x, context_y, context_z, observation_indicator): #, num_context_points):
        """Encodes the inputs into one representation.

        Args:
          TODO see if 1d inputs want a dummy dimension or not
          context_x: Tensor of size bs*observations
          context_y: Tensor of size bs*observations
          context_z: Tensor of size bs*observations
          observation_indicator: Tensor of size bs*observations. Provides the index to which observation each context_point belongs.

        Returns:
          representation: The encoded representation averaged over all context 
              points.
        """

        #one hot encode the channel context z, numeric value has no meaning, as opposed to time context
        context_z_onehot = tf.one_hot(context_z, self.n_channels) #n_datapoints x num_channels


        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([context_y,context_x, context_z_onehot], axis=-1)

        #Felix: Reshaping not necessary because don't have a batch dimension in encoder_input
        hidden = encoder_input

        # Pass through MLP
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            for i, size in enumerate(self._output_sizes[:-1]):
                hidden = tf.nn.relu(
                    tf.layers.dense(hidden, size, name="Encoder_layer_{}".format(i)))

                # Last layer without a ReLu
            hidden = tf.layers.dense(
                    hidden, self._output_sizes[-1], name="Encoder_layer_{}".format(i + 1))


        # Aggregator: take the sum over all points
        representation = tf.segment_sum(hidden, observation_indicator)

        return representation


class DeterministicDecoder(object):
  """The Decoder. """

  def __init__(self, output_sizes, n_channels):
    """CNP decoder.

    Args:
      output_sizes: An iterable containing the output sizes of the decoder MLP 
          as defined in `basic.Linear`.
    """
    self._output_sizes = output_sizes
    self.n_channels = n_channels

  def __call__(self, representation, target_x , target_z, observation_indicator):

    
    """Decodes the individual targets.
        Args:
      representation: The encoded representation of the context, batch_size x hidden_size
      target_x: The x locations for the target query, num_points x 1
      target_z: The channels for the target query, num_points x 1
      observation_indicator: The batch indices to which the points belong, num_points

    Returns:
      dist: A multivariate Gaussian over the target points.
      mu: The mean of the multivariate Gaussian.
      sigma: The standard deviation of the multivariate Gaussian.
    """

    #one hot encode the channel context z, numeric value has no meaning, as opposed to time context
    target_z_onehot = tf.one_hot(target_z, self.n_channels) #n_datapoints x num_channels

    #get the right representation for each query point
    target_representation = tf.gather(representation, observation_indicator) #sizes args: batch_size x hidden_size, n_datapoints => n_datapoints x hidden_size

    decoder_input = tf.concat([target_representation, target_x, target_z_onehot], axis=-1)

    hidden = decoder_input
     # Pass through MLP
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      for i, size in enumerate(self._output_sizes[:-1]):
        hidden = tf.nn.relu(
            tf.layers.dense(hidden, size, name="Decoder_layer_{}".format(i)))

      # Last layer without a ReLu
      hidden = tf.layers.dense(
          hidden, self._output_sizes[-1], name="Decoder_layer_{}".format(i + 1))

    # Bring back into original shape
    
    # Get the mean an the variance
    mu, log_sigma = tf.split(hidden, 2, axis=-1)

    # Bound the variance
    sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

    # Get the distribution
    #dist = tf.contrib.distributions.MultivariateNormalDiag(
    #    loc=mu, scale_diag=sigma)

    #return dist, mu, sigma
    return mu, sigma

class DeterministicDecoderDeprecated(object):
  """The Decoder. Not flexible the way i built it, rebuild it with inputs properly specified like the encoder, just with hidden_representation instead of y.
    Move the rest out of graph to numpy, more flexible and more control.
  """

  def __init__(self, output_sizes, n_channels):
    """CNP decoder.

    Args:
      output_sizes: An iterable containing the output sizes of the decoder MLP 
          as defined in `basic.Linear`.
    """
    self._output_sizes = output_sizes
    self.n_channels = n_channels

  def __call__(self, representation, target_x):#, num_total_points):
    """Decodes the individual targets.

    Args:
      representation: The encoded representation of the context
      target_x: The x locations for the target query
      num_total_points: The number of target points.

    Returns:
      dist: A multivariate Gaussian over the target points.
      mu: The mean of the multivariate Gaussian.
      sigma: The standard deviation of the multivariate Gaussian.
    """

    
    #create the context channels to append to target_x:
    channels = tf.range(self.n_channels) #shape n_channels
    context_z_onehot = tf.one_hot(channels, self.n_channels) #shape 44x44

    #target_x comes in shape batch_size x padded_length x 1

    #tiling and dimension expansion to make everything agree
    batch_size = tf.shape(target_x)[0]
    num_query_points = tf.shape(target_x)[1]
    target_x = tf.tile(target_x, [1,self.n_channels,1]) #batch_size x padded_length*n_channels x 1
    context_z_onehot = tf.expand_dims(context_z_onehot,0) # 1 x 44 x 44
    #context_z_onehot = tf.Print(context_z_onehot, [tf.shape(context_z_onehot)])
    target_z = tf.tile(context_z_onehot, [batch_size,num_query_points ,1]) # batch_size x padded_length*n_channels x 44

    #target_z = tf.Print(target_z,[tf.shape(target_z)])
    # Concatenate the representation and the target_x
    representation = tf.tile(
        tf.expand_dims(representation, axis=1), [1,num_query_points*self.n_channels, 1]) #insert dummy dim in representation, then tile
    input = tf.concat([representation, target_x, target_z], axis=-1) #concatenate along the last axis; shape batch_size x padded_length*n_channels x 44 + 1 + representation_dim

    #input = tf.Print(input, [tf.shape(input)], message ='input before reshaping')
    # Get the shapes of the input and reshape to parallelise across observations
    batch_size, _, filter_size = input.shape.as_list() #need this output for the filter_size set_shape
    batch_size = tf.shape(input)[0] #need this for the reshape - really don't get what the problem here is, one gives tensor one gives int or whatever
    #filter_size = tf.shape(input)[2]

    #hidden = tf.reshape(input, (batch_size * num_total_points, -1))

    flattened_length = batch_size * num_query_points*self.n_channels
    #flattened_length = tf.Print(flattened_length, [batch_size, num_query_points, self.n_channels], message = 'batchsize, num_points and n_channels for reshape size')
    hidden = tf.reshape(input, (flattened_length, -1))
    #hidden = tf.Print(hidden, [tf.shape(hidden)])

    hidden.set_shape((None, filter_size))
    hidden = tf.Print(hidden, [tf.where(tf.equal(hidden[0:5,129:],1))], message ='input to MLP slice', summarize = 20) #DEBUG see to which channels stuff belongs
    hidden = tf.Print(hidden, [hidden[0:100,128]], message ='input to MLP slice timepoint ind', summarize = 100)
    hidden = tf.Print(hidden, [tf.shape(hidden)], message ='input to MLP, reshaped for parallelization')

    # Pass through MLP
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      for i, size in enumerate(self._output_sizes[:-1]):
        hidden = tf.nn.relu(
            tf.layers.dense(hidden, size, name="Decoder_layer_{}".format(i)))

      # Last layer without a ReLu
      hidden = tf.layers.dense(
          hidden, self._output_sizes[-1], name="Decoder_layer_{}".format(i + 1))

    # Bring back into original shape
    hidden = tf.reshape(hidden, (batch_size, num_query_points*self.n_channels, -1))
    hidden = tf.reshape(hidden, (batch_size, num_query_points,self.n_channels, -1))

    # Get the mean an the variance
    mu, log_sigma = tf.split(hidden, 2, axis=-1)

    # Bound the variance
    sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

    # Get the distribution
    dist = tf.contrib.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)

    #return dist, mu, sigma
    return mu



class DeterministicModel(object):
  """The CNP model."""

  def __init__(self, encoder_output_sizes, decoder_output_sizes, n_channels, compute_loss = True):
    """Initialises the model.

    Args:
      encoder_output_sizes: An iterable containing the sizes of hidden layers of
          the encoder. The last one is the size of the representation r.
      decoder_output_sizes: An iterable containing the sizes of hidden layers of
          the decoder. The last element should correspond to the dimension of
          the y * 2 (it encodes both mean and variance concatenated)
      n_channels: Number of context channels (for one-hot)
      compute_loss = return Log_p of distribution fitted on context points
    """
    self._encoder = DeterministicEncoder(encoder_output_sizes, n_channels)
    self._decoder = DeterministicDecoder(decoder_output_sizes, n_channels)
    self.n_channels = n_channels
    self.compute_loss = compute_loss

  def __call__(self,context_x, context_y, context_z, context_observation_indicator, target_x , target_z, target_observation_indicator):
    """Returns the predicted mean and variance at the target points.

    Args:
      

    Returns:
      log_p: The log_probability of the target_y given the predicted
      distribution.
      mu: The mean of the predicted distribution.
      sigma: The variance of the predicted distribution.
    """

    
    # Pass context through the encoder
    representation = self._encoder(context_x, context_y, context_z, context_observation_indicator)

    #Pass target through decoder with representation
    mu_target, sigma_target = self._decoder(representation, target_x, target_z, target_observation_indicator)

    if self.compute_loss == False:
        return mu_target, sigma_target
    
    else:
        #Pass context through decoder with representation = on this we can compute the loss
        mu_reconstruction, sigma_reconstruction = self._decoder(representation, context_x, context_z, context_observation_indicator)

        #note: i fit one MGP for all data. What is the equivalent here. By fitting i really just decide how to calculate mu and sigma. This is what the encoder-decoder does here.
        #Futoma: vectorize the matrix, vector of all y_{timepoint,channel} --> do the same here, stack mu and sigma for each observation
        #order of stacking should be meaningless, just grab all that belong to the observation

        #will need a while loop as the observations have different numbers of points

        L = tf.zeros([0]) #Tensor to collect results
        N =  tf.shape(tf.unique(context_observation_indicator)[0])[0]#number of observations (observations= patients)

        """
        #test gathering etc for 1 observation without loop
        i =0
        #need to do
        #get indices of context_observation_indicator that are i
        i_indices = tf.where(tf.equal(context_observation_indicator,i))

        mu_i = tf.squeeze(tf.gather(mu_reconstruction, i_indices), axis =-1)
        sigma_i = tf.squeeze(tf.gather(sigma_reconstruction, i_indices), axis =-1)
        y_observed = tf.squeeze(tf.gather(context_y, i_indices), axis =-1)
        dist = tf.contrib.distributions.MultivariateNormalDiag(
                loc=mu_i, scale_diag=sigma_i)
        
        log_p = dist.log_prob(y_observed) #gives log_p for every point in y_observed
        total_log_p = tf.reduce_sum(log_p)
        mu_target = tf.Print(mu_target, [tf.shape(total_log_p)]) # 0 dim tensor
        """
        def cond(i,L):
            return i<N

        def body(i,L):

            
            #TODO is the order of channels meaningful in any way?
            i_indices = tf.where(tf.equal(context_observation_indicator,i))

            mu_i = tf.squeeze(tf.gather(mu_reconstruction, i_indices), axis =-1)
            sigma_i = tf.squeeze(tf.gather(sigma_reconstruction, i_indices), axis =-1)
            y_observed = tf.squeeze(tf.gather(context_y, i_indices), axis =-1)

            dist = tf.contrib.distributions.MultivariateNormalDiag(
                loc=mu_i, scale_diag=sigma_i)
            
            log_p = dist.log_prob(y_observed)
            total_log_p = tf.reduce_mean(log_p) #different number of points per observation, mean not sum
            total_log_p = tf.expand_dims(total_log_p, -1)
            L = tf.concat([L,total_log_p],0)
            #i = tf.Print(i, [i, log_p], message="welcome to while loop")
            return i+1,L

        i = tf.constant(0)
        i,L = tf.while_loop(cond,body,loop_vars=[i,L],
                shape_invariants=[i.get_shape(),tf.TensorShape([ None])])
        
        loss = L
        #mu_target = tf.Print(mu_target, [tf.shape(L)])


        return mu_target, sigma_target, loss