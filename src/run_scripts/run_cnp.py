import tensorflow as tf
import numpy as np
import sys
from time import time
from sacred import Experiment

from tempfile import NamedTemporaryFile
import traceback
import faulthandler

sys.path.insert(0, "/Users/felix/Coding/CNP-TCN/src")
sys.path.insert(0, '../../src')
from data_prepro.prepro_utils import load_dataset, flatten_batch, flatten_batch_query
from utils.util import count_parameters, num_grid_times_to_grid_times
from cnp.cnp import DeterministicModel, DeterministicEncoder, DeterministicDecoder


from IPython import embed

#TODO make this function more flexible for test val
#TODO labels would be nice

ex = Experiment('CNP-TCN')

@ex.config
def cnp_tcn_config():
    cnp_parameters = {
    'encoder_output_sizes' : [128, 128, 128, 128],
    'decoder_output_sizes' : [128, 128, 2], #2 for mean and variance
    'num_channels' : 44 #channels of the input data
    }
    tcn_parameters = {
    'n_channels': 40,
    'levels': 4, 
    'kernel_size': 4, 
    'dropout': 0.1,
    'n_classes' : 2 
    }

    training_parameters = {
    'batch_size' : 500,
    'epochs' : 30,
    'data_filepath' : '../data_prepro/mgp-rnn-datadump_labs_vitals_covs_na_thres_500_min_length_7_max_length_200_horizon_0_split_0.pkl',
    'learning_rate' : 0.000005    
    }



@ex.automain
def fit_cnp_tcn(cnp_parameters, tcn_parameters, training_parameters,  _rnd, _seed, _run):

    #load the data
    #outputs = simplify_time(data.times_test, data.ind_times_test), data.values_test, data.ind_lvs_test, data.num_rnn_grid_times_test, data.labels_test

    times_va, values_va, channels_va, num_grid_times_va, labels_va = load_dataset(training_parameters['data_filepath'], dataset='validation')
    times_tr, values_tr, channels_tr, num_grid_times_tr, labels_tr = load_dataset(training_parameters['data_filepath'], dataset='train')


    print('data loaded')
    rs = np.random.RandomState(_seed)
    #NOTE: first 4 elements are not structured by observations, completely flat. Observation info in output_observations. Maybe not smart.




    # Define the model
    model = DeterministicModel(cnp_parameters['encoder_output_sizes'], cnp_parameters['decoder_output_sizes'], cnp_parameters['num_channels'])
    encoder = DeterministicEncoder(cnp_parameters['encoder_output_sizes'], cnp_parameters['num_channels'])
    decoder = DeterministicDecoder(cnp_parameters['decoder_output_sizes'], cnp_parameters['num_channels'])
    print('model set up')
    #setup operations
    #placeholders
    x_input = tf.placeholder("float", [None,1]) #num_data_points 
    y_input = tf.placeholder("float", [None,1]) #num_data_points
    z_input = tf.placeholder(tf.int32, [None]) #num_data_points
    obs_ind = tf.placeholder(tf.int32, [None]) #num_data_points
    x_target  = tf.placeholder('float', [None, 1]) #num_query_points
    z_target = tf.placeholder(tf.int32, [None]) #num_query_points
    obs_ind_target = tf.placeholder(tf.int32, [None]) #num_query_points  


    mu, sigma, log_prob = model(x_input,y_input,z_input,obs_ind, x_target, z_target, obs_ind_target)
    loss = tf.reduce_mean(log_prob) * -1
    

    print('data stuff set up')
    #encoded_representation = encoder(x_input,y_input,z_input,obs_ind)
    #decoded_representation = decoder(encoded_representation, x_target, z_target, obs_ind_target)
    #decoded_representation_full = model(x_input,y_input,z_input,obs_ind, x_target, z_target, obs_ind_target)


    sess = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(training_parameters['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=global_step) 
    sess.run(tf.global_variables_initializer())
    print("Graph setup!")

    
    #tcn_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "temporal_conv_net/")

    count_parameters()


    #setup minibatch indices for training:
    batch_size = training_parameters['batch_size']
    Ntr = labels_tr.shape[0]

    starts = np.arange(0,Ntr,batch_size)
    ends = np.arange(batch_size,Ntr+1,batch_size)
    if ends[-1]<Ntr: 
        ends = np.append(ends,Ntr)
    num_batches = len(ends)

    #setup minibatch indices for validation (memory saving)
    Nva = labels_va.shape[0]
    va_starts = np.arange(0,Nva,batch_size)
    va_ends = np.arange(batch_size,Nva+1,batch_size)
    if va_ends[-1]<Nva: 
        va_ends = np.append(va_ends,Nva)


    #------------------- 
    # MGP training loop
    #-------------------
    saver = tf.train.Saver(max_to_keep = None)
    loss_list =[]
    total_batches = 0
    best_LL = 0
    for i in range(training_parameters['epochs']):

        #TODO fix completely
        #------------------- 
        # Train
        #------------------- 
        epoch_start = time()
        print("Starting epoch "+"{:d}".format(i))
        perm = rs.permutation(Ntr)
        batch = 0 
        for s,e in zip(starts,ends):

            batch_start = time()
            inds = perm[s:e]

            #prepare the data for the feed dict: model takes 1d tensors times, values, channels and obs_indicator as input
            #these are the training points for the encoder
            times_obs, values_obs, channels_obs, obs_inds_obs = flatten_batch(times_tr[inds], values_tr[inds], channels_tr[inds])
            #these are the query points for the decoder
            times_query, channels_query, obs_inds_query = flatten_batch_query(num_grid_times_tr[inds], cnp_parameters['num_channels'])
            

            feed_dict={x_input: times_obs ,y_input: values_obs, z_input: channels_obs,obs_ind: obs_inds_obs,
                x_target: times_query, z_target: channels_query, obs_ind_target: obs_inds_query    
                }
            



            try:        
                _loss, _ = sess.run([loss, train_op],feed_dict)
                
            except Exception as e:
                traceback.format_exc()
                print('Error occured in tensorflow during training:', e)
                #In addition dump more detailed traceback to txt file:
                with NamedTemporaryFile(suffix='.csv') as f:
                    faulthandler.dump_traceback(f)
                    _run.add_artifact(f.name, 'faulthandler_dump.csv')
                break

            print("Graph computed for batch {}, loss {}".format(batch, _loss))
            loss_list.append(_loss)
            batch += 1
    
    print(loss_list)
    losses = np.array(loss_list)
    embed()