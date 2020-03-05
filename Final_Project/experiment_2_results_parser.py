import numpy as np
import json
from Final_Project.reinforce_agent import ReinforceAgent
from Final_Project.dqn_agent import DQNAgent
from Final_Project.agent_trainer import trainer
import gym


def get_best_model(results):
    best_model_avg_perf = np.inf
    best_model = {}
    for result in results:
        if best_model_avg_perf > result['avg_num_of_episodes_to_solve']:
            best_model_avg_perf = result['avg_num_of_episodes_to_solve']
            best_model = result
    return best_model


with open('./results/reinforce_results_exp_2.json') as f:
    reinforce_results = json.load(f)

with open('./results/dqn_results_exp_2.json') as f:
    dqn_results = json.load(f)

print("Best REINFORCE model:")
print(get_best_model(reinforce_results))

print("Best DQN model:")
print(get_best_model(dqn_results))



env = gym.make('CartPole-v1')

# Best agents training and model saving:
best_reinforce_agent = ReinforceAgent(state_size=4, action_size=2, seed=123, gamma=1,
                                                     optimizer_learning_rate=0.01,
                                                     optimizer_weight_decay=0.005,
                                                     hidden_size=8, batch_norm=False,
                                                     dropout=False, activation='tanh',
                                                     init=('const', 0.1))
reinforce_scores, solved_in = trainer(env, best_reinforce_agent, num_episodes=5000, num_steps_in_episode=550,
                                                solved_checkpoint_name='best_reinforce_agent.pth', save_checkpoint=True,
                                                     verbose=True)


best_dqn_agent = DQNAgent(state_size=4, action_size=2, seed=123, update_every=10, gamma=1,
                                                 optimizer_learning_rate=0.005,
                                                 optimizer_weight_decay=0.0005,
                                                 batch_size=128, tau=1,
                                                 replay_buffer_size=1000000, eps_start=1, eps_end=0.1,
                                                 eps_decay=0.996, hidden_size=200, batch_norm=False,
                                                 dropout=False, activation='tanh', init=('pytorch_default', None))
dqn_scores, solved_in = trainer(env, best_dqn_agent, num_episodes=5000, num_steps_in_episode=550,
                                                solved_checkpoint_name='best_dqn_agent.pth', save_checkpoint=True,
                                                verbose=True)

import matplotlib.pyplot as plt

print(reinforce_scores)
print(dqn_scores)
plt.plot(reinforce_scores)
plt.plot(dqn_scores)
plt.legend(['REINFORCE score','DQN score'])
plt.xlabel('Episode number')
plt.show()