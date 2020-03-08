import os
import sys
import pandas as pd
from tempfile import NamedTemporaryFile
from collections import defaultdict
from sacred import Experiment
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
sys.path.insert(0, '../../src')
from sacred.observers import FileStorageObserver
from run_scripts.run_cnp_tcn_endtoend import ex as mgp_fit_experiment
import tensorflow as tf
ex = Experiment('hyperparameter_search_CNP_TCN_endtoend')
ex.observers.append(FileStorageObserver.create('cnp_tcn_hyperparameter_search_runs'))

@ex.config
def cfg():
    hyperparameter_space = {
        'learning_rate': ('Real', 0.0005, 0.005, 'log-uniform'),
        'n_channels': ('Integer', 15, 90), # = n_filters here
        'levels': ('Integer', 4,9), # causal dilation levels
        'kernel_size': ('Integer', 2,5),
        'batch_size': ('Integer', 10,300),
        'dropout': ('Real', 0.0, 0.1, 'uniform'),
        'l2_penalty': ('Real', 0.01, 100, 'log-uniform'), # not a standard lambda as loss is normalized by number of weights!
        'encoder_output_size': ('Integer', 100,512),
        'encoder_levels' : ('Integer', 3,6),
        'decoder_output_size': ('Integer', 100,512),
        'decoder_levels': ('Integer', 3,5),
        'log_prob_fraction': ('Real', 0.00001,0.9)

    }
    # These parameters will be passed to the mgp_rnn_experiment
    overrides = {
        'epochs': 100,
    }
    n_random_starts = 10
    n_calls = 20

@ex.capture
def build_search_space(hyperparameter_space):
    space = []
    for name, subspace in hyperparameter_space.items():
        parameter_type = subspace[0]
        parameters = subspace[1:]
        if parameter_type == 'Real':
            space.append(Real(*parameters, name=name))
        elif parameter_type == 'Integer':
            space.append(Integer(*parameters, name=name))
        elif parameter_type == 'Categorical':
            space.append(Categorical(*parameters, name=name))
        else:
            ValueError('{} is not a valid parameter_type'.format(parameter_type))
    return space


@ex.main
def search_hyperparameter_space(n_random_starts, n_calls, overrides, _rnd, _run):
    search_space = build_search_space()
    parameter_name = [var.name for var in search_space]
    if len(_run.observers) > 0:
        run_dir = _run.observers[0].dir
        mgp_fit_experiment.observers.append(FileStorageObserver.create(os.path.join(run_dir, 'mgp_rnn_runs')))

    @use_named_args(search_space)
    def objective(**params):
        tf.reset_default_graph()
        params.update(overrides)
        print('Running mgp_fit experiment with parameters: {}'.format(params))
        try:
            run = mgp_fit_experiment.run(config_updates=params)
            return -run.result['Best Validation AUPRC'] #since we use gp_minimize we need to rephrase maximization to minimization
        except Exception as e:
            print('An exception occured in mgp_rnn_fit:', e)
            return 0.

    res_gp = gp_minimize(objective, search_space, n_calls=n_calls, n_random_starts=n_random_starts, random_state=_rnd)
    
    parameter_iterations = defaultdict(list)
    for parameter_evaluation, objective_value in zip(res_gp.x_iters, res_gp.func_vals):
        for name, value in zip(parameter_name, parameter_evaluation):
            parameter_iterations[name].append(value)
        parameter_iterations['val_auprc'].append(objective_value)

    with NamedTemporaryFile(suffix='.csv') as f:
        df = pd.DataFrame.from_dict(parameter_iterations)
        df.to_csv(f.name)
        #f.flush()
        _run.add_artifact(f.name, 'parameter_evaluations.csv')

    best_parameters = {variable.name: value for variable, value in zip(search_space, res_gp.x)}
    return {'Best score': res_gp.fun, 'best_parameters': best_parameters}

if __name__ == '__main__':
    ex.run_commandline()
