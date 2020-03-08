import tensorflow as tf
import numpy as np
import sys
import os
from time import time
from sacred import Experiment
from sklearn.metrics import roc_auc_score, average_precision_score


from tempfile import NamedTemporaryFile
import traceback
import faulthandler

sys.path.insert(0, "/Users/felix/Coding/CNP-TCN/src")
sys.path.insert(0, '../../src')
from data_prepro.prepro_utils import load_dataset, flatten_batch, flatten_batch_query
from utils.util import count_parameters, num_grid_times_to_grid_times
from cnp.cnp import DeterministicModel, DeterministicEncoder, DeterministicDecoder
from models.cnp_tcn import CNPTCN


#TODO labels would be nice

ex = Experiment('CNP-TCN')

@ex.config
def cnp_tcn_config():
    #makes feeding params from hyperparameter optimization easier, don't need to access dicts directly
    encoder_output_size = 128 #[128, 128, 128, 128] 
    decoder_output_size = 128 #[128, 128, 2]
    decoder_levels = 3
    encoder_levels = 4
    n_channels = 40
    levels = 4
    kernel_size = 4
    dropout = 0.1
    batch_size = 500
    epochs = 100
    learning_rate = 0.0001
    l2_penalty = 100
    log_prob_fraction = 0.00001


    tcn_parameters = {
    'n_channels': n_channels,
    'levels': levels, 
    'kernel_size': kernel_size, 
    'dropout': dropout,
    'n_classes' : 2 
    }

    cnp_parameters = {
    'encoder_output_size' : encoder_output_size,
    'decoder_output_size' : encoder_output_size,
    'decoder_levels' : decoder_levels,
    'encoder_levels' : encoder_levels,
    'num_channels' : 44 #channels of the input data
    }

    training_parameters = {
    'batch_size' : batch_size,
    'epochs' : epochs,
    'data_filepath' : '../data_prepro/mgp-rnn-datadump_labs_vitals_covs_na_thres_500_min_length_7_max_length_200_horizon_0_split_0.pkl',
    'learning_rate' : learning_rate,
    'l2_penalty' : l2_penalty,
    'log_prob_fraction': log_prob_fraction #loss is fraction*log_prob_loss  + (1-fraction)*classifier_loss    
    }



@ex.automain
def fit_cnp_tcn(cnp_parameters, tcn_parameters, training_parameters,  _rnd, _seed, _run):
     #Path to save model:
    if len(_run.observers) > 0:
         checkpoint_path = os.path.join(_run.observers[0].dir, 'model_checkpoints')
    else:
        checkpoint_path = 'model_checkpoints'

    #load the data
    #outputs = simplify_time(data.times_test, data.ind_times_test), data.values_test, data.ind_lvs_test, data.num_rnn_grid_times_test, data.labels_test

    times_va, values_va, channels_va, num_grid_times_va, labels_va = load_dataset(training_parameters['data_filepath'], dataset='validation')
    times_tr, values_tr, channels_tr, num_grid_times_tr, labels_tr = load_dataset(training_parameters['data_filepath'], dataset='train')




    print('data loaded')

    #Get class imbalance (for weighted loss):
    case_prev = labels_tr.sum()/float(len(labels_tr)) #get prevalence of cases in train dataset
    class_imb = 1/case_prev #class imbalance to use as class weight if losstype='weighted'

    ##test_freq     = Ntr/batch_size #eval on test set after this many batches
    test_freq = 5#int(values_tr.shape[0]/training_parameters['batch_size'] / 4)


    rs = np.random.RandomState(_seed)
    tf.set_random_seed(_seed)


    # Define the model
    model = CNPTCN(tcn_parameters, cnp_parameters,class_imb,training_parameters['l2_penalty'])
    #model = DeterministicModel(cnp_parameters['encoder_output_sizes'], cnp_parameters['decoder_output_sizes'], cnp_parameters['num_channels'])

    #encoder = DeterministicEncoder(cnp_parameters['encoder_output_sizes'], cnp_parameters['num_channels'])
    #decoder = DeterministicDecoder(cnp_parameters['decoder_output_sizes'], cnp_parameters['num_channels'])
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
    labels = tf.placeholder(tf.int32, [None]) #batch_size
    labels_one_hot =  tf.one_hot(labels, tcn_parameters['n_classes']) #batch_size x 2
    is_training = tf.placeholder("bool")



    #mu, sigma, log_prob = model.cnp(x_input,y_input,z_input,obs_ind, x_target, z_target, obs_ind_target)
    logits, cnp_log_prob, tcn_loss = model(x_input,y_input,z_input,obs_ind, x_target, z_target, obs_ind_target, labels_one_hot, is_training)
    probs = model.compute_probs(logits)

    #Z, log_prob = model._get_cnp_output(x_input,y_input,z_input,obs_ind, x_target, z_target, obs_ind_target)
    loss = tf.reduce_mean(cnp_log_prob) * -1

    #Idea to test: balance training of cnp and tcn, find sweet spot of mgp-like behaviour of cnp and best classifier performance.
    #Maybe is dumb idea and should focus on classifier only

    #get two weights that fulfill ratio condition and sum to 1

    complete_loss = training_parameters['log_prob_fraction']*loss + (1-training_parameters['log_prob_fraction'])* tcn_loss



    sess = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(training_parameters['learning_rate'])
    train_op = optimizer.minimize(complete_loss, global_step=global_step) 
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
    best_val =0
    for i in range(training_parameters['epochs']):

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
                x_target: times_query, z_target: channels_query, obs_ind_target: obs_inds_query, labels: labels_tr[inds],
                is_training: True    
                }
            
            try:        
                loss_,logits_, probs_, _ = sess.run([complete_loss,logits, probs, train_op],feed_dict)          
            except Exception as e:
                traceback.format_exc()
                print('Error occured in tensorflow during training:', e)
                #In addition dump more detailed traceback to txt file:
                with NamedTemporaryFile(suffix='.csv') as f:
                    faulthandler.dump_traceback(f)
                    _run.add_artifact(f.name, 'faulthandler_dump.csv')
                break

            print("Batch "+"{:d}".format(batch)+"/"+"{:d}".format(num_batches)+\
                  ", took: "+"{:.3f}".format(time()-batch_start)+", loss: "+"{:.5f}".format(loss_))
            sys.stdout.flush()
            batch += 1; total_batches += 1
    
            if total_batches % test_freq == 0: #Check val set every so often for early stopping

                #--------------------------
                # Validate and write scores
                #-------------------------- 

                print('--> Entering validation step...')
                val_t = time()
                #Batch-wise Validation Phase:
                va_probs_tot = np.array([])
                va_perm = rs.permutation(Nva)

                va_labels_tot = labels_va[va_perm]
               


                perm = rs.permutation(Ntr)
                batch = 0 
                for v_s,v_e in zip(va_starts,va_ends):

                    batch_start = time()
                    #inds = perm[s:e]
                    va_inds = va_perm[v_s:v_e]

                    #prepare the data for the feed dict: model takes 1d tensors times, values, channels and obs_indicator as input
                    #these are the training points for the encoder
                    times_obs, values_obs, channels_obs, obs_inds_obs = flatten_batch(times_va[va_inds], values_va[va_inds], channels_va[va_inds])
                    #these are the query points for the decoder
                    times_query, channels_query, obs_inds_query = flatten_batch_query(num_grid_times_va[va_inds], cnp_parameters['num_channels'])


                    feed_dict={x_input: times_obs ,y_input: values_obs, z_input: channels_obs,obs_ind: obs_inds_obs,
                        x_target: times_query, z_target: channels_query, obs_ind_target: obs_inds_query, labels: labels_va[va_inds],
                        is_training: False    
                        }
                    

                    try:        
                        va_loss_,va_logits_, va_probs_ = sess.run([complete_loss,logits, probs],feed_dict)
                        
                    except Exception as e:
                        traceback.format_exc()
                        print('Error occured in tensorflow during training:', e)
                        #In addition dump more detailed traceback to txt file:
                        with NamedTemporaryFile(suffix='.csv') as f:
                            faulthandler.dump_traceback(f)
                            _run.add_artifact(f.name, 'faulthandler_dump.csv')
                        break
                    va_probs_tot = np.concatenate([va_probs_tot, va_probs_])

                va_auc = roc_auc_score(va_labels_tot, va_probs_tot)
                va_prc = average_precision_score(va_labels_tot, va_probs_tot)   
                best_val = max(va_prc, best_val)
                print("Epoch "+str(i)+", seen "+str(total_batches)+" total batches. Validating Took "+\
                    "{:.2f}".format(time()-val_t)+\
                    ". OOS, "+str(0)+" hours back: Loss: "+"{:.5f}".format(va_loss_)+ \
                    ", AUC: {:.5f}".format(va_auc)+", AUPR: "+"{:.5f}".format(va_prc))
                _run.log_scalar('train_loss', va_loss_, total_batches)
                _run.log_scalar('val_auprc', va_prc, total_batches)
                #_run.log_scalar('lenghts', length_, total_batches)
                sys.stdout.flush()    

                #create a folder and put model checkpoints there
                saver.save(sess, checkpoint_path + "/epoch_{}".format(i), global_step=total_batches)
        print("Finishing epoch "+"{:d}".format(i)+", took "+\
                "{:.3f}".format(time()-epoch_start))

        ### Takes about ~1-2 secs per batch of 50 at these settings, so a few minutes each epoch
        ### Should converge reasonably quickly on this toy example with these settings in a few epochs
    return {'Best Validation AUPRC': best_val}