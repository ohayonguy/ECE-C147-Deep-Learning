import Final_Project.dqn_agent as dqn
import Final_Project.reinforce_agent as reinforce_agent
from Final_Project.agent_trainer import trainer
import gym
import matplotlib.pyplot as plt
'''
Ideas:
The DQN and REINFORCE neural network should be the same?, except for the softmax layer at the end. Try to explain why it
might be a good idea if those networks are of the same architecture.

Transfer learning: maybe learn a DQN agent, and then transform it to a policy gradient to extract the best policy and see
how many extra steps it takes to learn from there.
'''
env = gym.make('CartPole-v1')

dqn_agent = dqn.DQNAgent(state_size=4, action_size=2, seed=123, update_every=10, tau=1, gamma=1,
                 optimizer_learning_rate=0.005, optimizer_weight_decay=0.005, batch_size=128,
                 replay_buffer_size=1000000, eps_start=0.99, eps_end=0.1, eps_decay=0.99)
scores = trainer(env, dqn_agent, solved_checkpoint_name='dqn_agent_solved.pth')
reinforce_agent = reinforce_agent.ReinforceAgent(state_size=4, action_size=2, seed=123, gamma=1, optimizer_learning_rate=0.005,
                 optimizer_weight_decay=0.005)
#scores = trainer(env, reinforce_agent, solved_checkpoint_name='reinforce_agent_solved.pth')

env.close()

plt.plot(scores)
plt.show()
