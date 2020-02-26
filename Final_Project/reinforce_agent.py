import argparse
import gym
import numpy as np
from itertools import count
import random
import torch
from Final_Project.models import Policy
import torch.optim as optim
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReinforceAgent():
    '''
    https://medium.com/@aminamollaysa/policy-gradients-and-log-derivative-trick-4aad962e43e0
    https://mcneela.github.io/math/2018/04/18/A-Tutorial-on-the-REINFORCE-Algorithm.html
    https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#reinforce
    https://www.youtube.com/watch?v=bRfUxQs6xIM&list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs&index=10
    '''
    def __init__(self, state_size, action_size, gamma=0.99, seed=543, optimizer_learning_rate=0.005,
                 optimizer_weight_decay=0.005, hidden_size=128, batch_norm=False, dropout=False,
                 activation='none', init=('const', 1)):
        self.gamma = gamma

        self.policy_network = Policy(state_size, action_size, seed, hidden_size, batch_norm, dropout, activation, init).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=optimizer_learning_rate,
                                    weight_decay=optimizer_weight_decay)
        self.saved_log_probs = []
        self.episode_rewards = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_network.eval()
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def learn(self, last_step_in_episode, done, reward, **kwargs):
        self.policy_network.train()
        self.episode_rewards.append(reward)
        if last_step_in_episode or done:
            eps = np.finfo(np.float32).eps.item()
            R = 0
            policy_loss = []
            returns = []
            for r in reversed(self.episode_rewards):

                # R is the discounted total reward of the episode, from its end up until step i.
                R = r + self.gamma * R

                # returns is an array in which returns[i] is the total discounted reward from step i onwards.
                returns.insert(0, R)

            returns = torch.tensor(returns).to(device)
            # Reduce variance
            returns = (returns - returns.mean()) / (returns.std() + eps)
            for log_prob, R in zip(self.saved_log_probs, returns):
                policy_loss.append(-log_prob * R)
            self.optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
            del self.episode_rewards[:]
            del self.saved_log_probs[:]

    def get_agent_model(self):
        return self.policy_network
