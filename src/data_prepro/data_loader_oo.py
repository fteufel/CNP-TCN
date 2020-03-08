import numpy as np
import pickle
from IPython import embed

#utiliy function to fit data into memory (one outlier patient has 11k observation values, remove outliers by cut-off) 
def mask_large_samples(data, thres, obs_min, static=None, return_mask=False):
    result_data = []
    n = len(data) #number of data views of compact format (values, times, indices, ..)
    mask = data[8] <= thres
    min_mask = data[8] >= obs_min #mask patients with less than n_mc_smps many num_obs_values
    print('-> {} patients have less than {} observation values'.format(np.sum(~min_mask),obs_min))
    mask = np.logical_and(mask, min_mask)
    print('---> Removing {} patients'.format(np.sum(~mask)))
    for i in np.arange(n):
        result_data.append(data[i][mask])
    if static is not None:
        result_static = static[mask]
        if return_mask:
            return result_data, result_static, mask 
        else:
            return result_data, result_static
    else:
        if return_mask:
            return result_data, mask
        else:
            return result_data


class DataContainer():
    """""
    Container class for the input data. Loads the data from a .pkl at filepath, and processes it according to data_sources
    masks large samples before returning the data. TODO: has hardcoded configs right now
    """""

    def __init__(self, filepath, data_sources):
        #Define path to dataset
        datapath = filepath

        #Configs:
        #data_sources = ['labs','vitals','covs']
        num_obs_thres = 10000
        n_mc_smps = 10 
        obs_min = np.max([10, n_mc_smps]) #remove the samples with less than 10 observation values, Futoma's lanczos impl fails when mc_smps > num_obs!
        
        #Load full dataset from pickle
        full_dataset = pickle.load( open( datapath, "rb" ))

        #Unpack dataset:
        variables = full_dataset[0] #list of all used variables
        train_data,validation_data,test_data = full_dataset[1:4]
        #if 'covs' in data_sources:
        train_static_data, validation_static_data, test_static_data = full_dataset[4:7]

        #Masking step (removing outlier patients with too many observations (memory issue) or too few)
        train_data, train_static_data = mask_large_samples(train_data, num_obs_thres, obs_min, static=train_static_data)
        validation_data, validation_static_data = mask_large_samples(validation_data, num_obs_thres, obs_min, static=validation_static_data)
        test_data, test_static_data = mask_large_samples(test_data, num_obs_thres, obs_min, static=test_static_data)

        #data as array for interfacing with evaluation functions
        self.train_data = train_data
        self.test_data = test_data
        self.validation_data = validation_data

        #Checking data split counts
        self.M = len(variables)  
        self.Ntr = len(train_data[0])
        self.Nva = len(validation_data[0])
        self.Ntest = len(test_data[0])
        self.n_covs = 0

        
        #Unpacking compact format of data splits (Train and Val for now)
        self.values_tr = train_data[0]; self.values_va = validation_data[0]; self.values_test = test_data[0]
        self.times_tr = train_data[1]; self.times_va = validation_data[1]; self.times_test = test_data[1]
        self.ind_lvs_tr = train_data[2]; self.ind_lvs_va = validation_data[2]; self.ind_lvs_test = test_data[2]
        self.ind_times_tr = train_data[3]; self.ind_times_va = validation_data[3]; self.ind_times_test = test_data[3]
        self.labels_tr = train_data[4]; self.labels_va = validation_data[4]; self.labels_test = test_data[4]
        self.num_rnn_grid_times_tr = train_data[5]; self.num_rnn_grid_times_va = validation_data[5]; self.num_rnn_grid_times_test = test_data[5]
        self.rnn_grid_times_tr = train_data[6]; self.rnn_grid_times_va = validation_data[6]; self.rnn_grid_times_test = test_data[6]
        self.num_obs_times_tr = train_data[7]; self.num_obs_times_va = validation_data[7]; self.num_obs_times_test = test_data[7]
        self.num_obs_values_tr = train_data[8]; self.num_obs_values_va = validation_data[8]; self.num_obs_values_test = test_data[8]
        self.onset_times_tr = train_data[9]; self.onset_times_va = validation_data[9]; self.onset_times_test = test_data[9]
        if 'covs' in data_sources:
            self.covs_tr = train_static_data
            self.n_covs = train_static_data.shape[1]
            self.covs_va = validation_static_data #; covs_te = test_static_data
            self.covs_test = test_static_data    

        #Get class imbalance (for weighted loss):
        #case_prev = labels_tr.sum()/float(len(labels_tr)) #get prevalence of cases in train dataset
        #class_imb = 1/case_prev #class imbalance to use as class weight if losstype='weighted'

        





