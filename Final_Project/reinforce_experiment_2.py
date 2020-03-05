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

results = []
reinforce_json_file = open('best_reinforce_results.json', 'w')
k = 0
gammas = gammas = [0.92, 0.96, 1]
learning_rates = [0.5, 0.05, 0.005, 0.0005, 0.01]
hidden_sizes = [8, 10, 12, 14, 16]
weight_decayes = [0.5, 0.05, 0.005, 0.0005]
bn = False
activation = 'tanh'
initialization = ('const', 0.1)
for gamma in gammas:
    for optimizer_learning_rate in learning_rates:
        for hidden_size in hidden_sizes:
            for weight_decay in weight_decayes:
                num_episodes_to_solve_list = []
                solved = True
                for i in range(num_of_solutions_to_examine):
                    best_reinforce_agent = ReinforceAgent(state_size=4, action_size=2, seed=123, gamma=gamma,
                                                     optimizer_learning_rate=optimizer_learning_rate,
                                                     optimizer_weight_decay=weight_decay,
                                                     hidden_size=hidden_size, batch_norm=bn,
                                                     dropout=False, activation=activation,
                                                     init=initialization)
                    scores, solved_in = trainer(env, best_reinforce_agent, num_episodes=5000, num_steps_in_episode=550,
                                                solved_checkpoint_name='reinforce_agent_solved.pth')
                    num_episodes_to_solve_list.append(solved_in)
                    if solved_in == 5000:
                        solved = False
                        break
                line_to_append = {'gamma': gamma, 'optimizer_learning_rate': optimizer_learning_rate,
                                'optimizer_weight_decay': weight_decay,
                                'hidden_size': hidden_size,
                                'avg_num_of_episodes_to_solve': np.mean(num_episodes_to_solve_list),
                                'solved': solved}
                results.append(line_to_append)
                print(line_to_append)
                k += 1
                #print('\rIteration: {}'.format(k), end="")

json.dump(results, reinforce_json_file)
reinforce_json_file.close()