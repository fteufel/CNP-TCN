import numpy as np
import sys
from .data_loader_oo import DataContainer


#turn ind_times_tr and times_tr into timepoint per datapoint tensor


def simplify_time(times, ind_times):

    output = []
    for i in range(times.shape[0]):
        simplified_times = times[i][ind_times[i]]
        output.append(simplified_times)
    return np.array(output)


def load_dataset(filepath, dataset = 'train', data_sources =['labs','vitals','covs']):
    assert dataset in ['train','test','validation'], 'No valid dataset specified'
    data = DataContainer(filepath, data_sources)
    if dataset == 'train':
        outputs = simplify_time(data.times_tr, data.ind_times_tr), data.values_tr, data.ind_lvs_tr, data.num_rnn_grid_times_tr, data.labels_tr
    if dataset == 'validation':
        outputs = simplify_time(data.times_va, data.ind_times_va), data.values_va, data.ind_lvs_va, data.num_rnn_grid_times_va, data.labels_va
    if dataset == 'test':
        outputs = simplify_time(data.times_test, data.ind_times_test), data.values_test, data.ind_lvs_test, data.num_rnn_grid_times_test, data.labels_test
    return outputs

    
def flatten_batch(times, values, ind_channels):
    assert times.shape == ind_channels.shape  == values.shape
    
    output_times = np.empty(0)
    output_values = np.empty(0)
    output_channels = np.empty(0)
    output_observations = np.empty(0)
    #iterate over all observations
    for i in range(times.shape[0]):
        #create the observation indicator array
        obs_indicator = np.ones(values[i].shape[0]) * i

        #flatten data
        output_values = np.append(output_values, values[i])
        output_channels = np.append(output_channels, ind_channels[i])
        output_times = np.append(output_times, times[i])
        output_observations = np.append(output_observations, obs_indicator)
        
    output_values = np.expand_dims(output_values,-1) # num_total_datapoints x 1
    output_times = np.expand_dims(output_times,-1)
    return output_times, output_values, output_channels.astype(int), output_observations.astype(int)

def flatten_batch_query(num_grid_times, num_channels):

    output_times = np.empty(0)
    output_channels = np.empty(0)
    output_observations = np.empty(0)

    for i in range(num_grid_times.shape[0]):

        #make the grid times
        grid_times = np.arange(num_grid_times[i])
        channels = np.arange(num_channels)

        # repeat and time so that every (timepoint, channel) pair is a datapoint
        grid_times = np.repeat(grid_times, num_channels)
        channels = np.tile(channels, num_grid_times[i])

        obs_indicator = np.ones(channels.shape[0]) * i

        output_observations = np.append(output_observations, obs_indicator)
        output_channels = np.append(output_channels, channels)
        output_times = np.append(output_times, grid_times)

    output_times = np.expand_dims(output_times,-1)

    return output_times, output_channels.astype(int), output_observations.astype(int)