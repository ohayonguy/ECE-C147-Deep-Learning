import json

file = './results/dqn_results_exp_1.json'
with open(file) as f:
    results = json.load(f)


batch_norms = [False, True]
initializations = [['const', 0.1], ['const', 30],
                   ['uniform', 0.1], ['uniform', 30],
                   ['xavier_uniform', None], ['pytorch_default', None]]
activations = ['none', 'relu', 'leakyrelu', 'tanh']
dropout_choices = [False]
hidden_sizes = [1, 10, 20, 80, 256]

def get_best_3_models(hyper_param_name, list_of_possible_features, results):
    best_models = {}
    for hyper_param in list_of_possible_features:
        best_models[hyper_param_name + '_' + str(hyper_param)] = {}
        filtered_results = list(filter(lambda k: k[hyper_param_name] == hyper_param, results))
        sorted_by_performance = sorted(filtered_results, key=lambda k: k['avg_num_of_episodes_to_solve'])
        best_models[hyper_param_name + '_' + str(hyper_param)]['1st'] = sorted_by_performance[0]
        best_models[hyper_param_name + '_' + str(hyper_param)]['2nd'] = sorted_by_performance[1]
        best_models[hyper_param_name + '_' + str(hyper_param)]['3rd'] = sorted_by_performance[2]
    return best_models

best_models_hidden_size =  get_best_3_models('hidden_size', hidden_sizes, results)
best_models_activation = get_best_3_models('activation', activations, results)
best_models_bn = get_best_3_models('batch_norm', batch_norms, results)
best_models_init = get_best_3_models('init', initializations, results)

print("best for hidden size:")
print(best_models_hidden_size)

print("best for activation:")
print(best_models_activation)

print("best for batch norm:")
print(best_models_bn)

print("best for initialization:")
print(best_models_init)
'''
best_models = {}
# Iterate hidden layer size. For each size, get best 3 models.
hidden_sizes = [1, 10, 20, 80, 256]
for hidden_size in hidden_sizes:
    best_models['hidden_size_'+str(hidden_size)] = {}
    filtered_results = list(filter(lambda k: k['hidden_size'] == hidden_size, results))
    sorted_by_performance = sorted(filtered_results, key=lambda k: k['avg_num_of_episodes_to_solve'])
    best_models['hidden_size_'+str(hidden_size)]['1st'] = sorted_by_performance[0]
    best_models['hidden_size_'+str(hidden_size)]['2nd'] = sorted_by_performance[1]
    best_models['hidden_size_'+str(hidden_size)]['3rd'] = sorted_by_performance[2]

activations = ['none', 'relu', 'leakyrelu', 'tanh']
for activation in activations:
    best_models['activation_'+str(activation)] = {}
    filtered_results = list(filter(lambda k: k['activation'] == activation, results))
    sorted_by_performance = sorted(filtered_results, key=lambda k: k['avg_num_of_episodes_to_solve'])
    best_models['activation_'+str(activation)]['1st'] = sorted_by_performance[0]
    best_models['activation_'+str(activation)] = sorted_by_performance[1]
    best_models['activation_'+str(activation)] = sorted_by_performance[2]
print(best_models)

'''

'''
best_dqn_file = 'results/dqn_results_exp_2.json'
best_reinforce_file = 'results/reinforce_results_exp_2.json'

with open(best_dqn_file) as f:
    dqn_results = json.load(f)

with open(best_reinforce_file) as f:
    reinforce_results = json.load(f)

import numpy as np

best_reinforce_parameters = None
best_reinforce_perf = np.inf
for result in reinforce_results:
    if result['avg_num_of_episodes_to_solve'] < best_reinforce_perf:
        best_reinforce_perf = result['avg_num_of_episodes_to_solve']
        best_reinforce_parameters = result

best_dqn_parameters = None
best_dqn_perf = np.inf
for result in dqn_results:
    if result['avg_num_of_episodes_to_solve'] < best_dqn_perf:
        best_dqn_perf = result['avg_num_of_episodes_to_solve']
        best_dqn_parameters = result

print("Best reinforce model hyperparameters:")
print(best_reinforce_parameters)

print("Best dqn model hyperparameters:")
print(best_dqn_parameters)
'''