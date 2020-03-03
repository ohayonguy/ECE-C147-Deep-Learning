import Final_Project.dqn_agent as dqn
from Final_Project.reinforce_agent import ReinforceAgent
from Final_Project.agent_trainer import trainer
import gym
import json
import matplotlib.pyplot as plt
import torch
import numpy as np

env = gym.make('CartPole-v1')

num_of_solutions_to_examine = 5

batch_norms = [False, True]
initializations = [('const', 0.1), ('const', 30),
                   ('uniform', 0.1), ('uniform', 30),
                   ('xavier_uniform', None)]
activations = ['none', 'relu', 'leakyrelu', 'tanh']
dropout_choices = [False]
hidden_sizes = [1, 10, 20, 80, 256]

results = []
dqn_results_file = open('dqn_results.txt', 'w')
dqn_json_file = open('dqn_results.json', 'w')
k = 0
for bn in batch_norms:
    for initialization in initializations:
        for activation in activations:
            for dropout in dropout_choices:
                for hidden_size in hidden_sizes:
                    num_episodes_to_solve_list = []
                    solved = True
                    for i in range(num_of_solutions_to_examine):
                        dqn_agent = dqn.DQNAgent(state_size=4, action_size=2, seed=123, update_every=10, gamma=1,
                                                 optimizer_learning_rate=0.01, optimizer_weight_decay=0.005,
                                                 batch_size=128, tau=1,
                                                 replay_buffer_size=1000000, eps_start=1, eps_end=0.1,
                                                 eps_decay=0.996, hidden_size=hidden_size, batch_norm=bn,
                                                 dropout=dropout, activation=activation, init=initialization)

                        scores, solved_in = trainer(env, dqn_agent, num_episodes=5000, num_steps_in_episode=550,
                                                    solved_checkpoint_name='dqn_agent_solved.pth')
                        num_episodes_to_solve_list.append(solved_in)
                        if solved_in == 5000:
                            solved = False
                            break
                    line_to_write = "Init: " + str(initialization)+". Activation: " + str(activation) + ". Batch norm: " + str(bn) +". Dropout: " + str(dropout)+". Hidden size: " + str(hidden_size)+". " + "Average num of episodes to solve: " + str(np.mean(num_episodes_to_solve_list)) +". Solved: " + str(solved)
                    dqn_results_file.write(line_to_write)
                    results.append({'init': initialization, 'activation': activation, 'batch_norm': bn,
                                    'dropout': dropout, 'hidden_size': hidden_size,
                                    'avg_num_of_episodes_to_solve': np.mean(num_episodes_to_solve_list),
                                    'solved': solved})
                    k += 1
                    print('\rIteration: {}'.format(k), end="")

json.dump(results, dqn_json_file)
dqn_results_file.close()
dqn_json_file.close()