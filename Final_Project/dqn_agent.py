import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from Final_Project.models import QNetwork
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent():
    def __init__(self, state_size, action_size, seed=123, update_every=10, tau=1, gamma=1,
                 optimizer_learning_rate=0.005, optimizer_weight_decay=0.99, batch_size=1024,
                 replay_buffer_size=1000000, eps_start=0.99, eps_end=0.1, eps_decay=0.99,
                 hidden_size=128, batch_norm=False, dropout=False, activation='none', init=('const', 1)):
        '''
        Initialize the DQN agent.
        @param state_size: the size of the observed state
        @type state_size: numpy array with shape (S,)

        @param action_size: the size of the action space
        @type action_size: numpy array with shape (A,)

        @param seed: random seed
        @type seed: non-negative int

        @param update_every: defines in between how many steps we update the target network.
        @type update_every: non-negative int

        @param tau: the target network's update is softly updated by the parameter tau. tau=small means that the weights
        are softly updated in the target network. tau=large means that the weights are hardly updated in the target
        network.
        @type tau: float between 0 and 1

        @param gamma: discount factor
        @type gamma: non-negative float

        @param optimizer_learning_rate: the learning rate for the neural network optimizer
        @type optimizer_learning_rate: non-negative float

        @param optimizer_weight_decay: the weight decay parameter for the networks weights (L2 penalty).
        @type optimizer_weight_decay: float between 0 and 1

        @param batch_size: the batch size from the replay buffer to calculate the q network's gradient.
        @type batch_size: non-negative int

        @param replay_buffer_size: the size of the replay buffer
        @type replay_buffer_size: non-negative int

        @param eps_start: the exploration parameter to start with
        @type replay_buffer_size: float between 0 and 1

        @param eps_end: the exploration parameter to end with
        @type replay_buffer_size: float between 0 and 1

        @param eps_end: the exploration parameter decay
        @type replay_buffer_size: float between 0 and 1
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.update_every = update_every
        self.tau = tau
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_size, batch_norm, dropout,
                                       activation, init).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_size, batch_norm, dropout,
                                        activation, init).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=optimizer_learning_rate,
                                    weight_decay=optimizer_weight_decay)

        # Replay buffer
        self.memory = ReplayBuffer(action_size, replay_buffer_size, batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def select_action(self, state):
        '''
        Select the next action given a state.

        @param state: the current state
        @type state: numpy array with shape (S,)

        @param last_step_in_episode: True if this is the last action selection in the episode. False otherwise.
        @type state: boolean

        @return: the action selection
        @rtype: an integer between 1 and A
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, state, action, reward, next_state, done, last_step_in_episode):
        '''
        Make a bellman update step every self.update_every steps.

        @param state: the current state
        @type state: numpy array with shape (S,)

        @param action: the action to perform
        @type action: numpy array with shape (A,)

        @param reward: the immediate reward returned after performing the action in the current state
        @type reward: float

        @param next_state: the next state following the current state, after performing the action
        @type next_state: numpy array with shape (S,)

        @param done: specifies if the episode is finished (whether the next state is a terminal state or not)
        @type done: boolean
        '''
        if last_step_in_episode or done:
            self.eps = max(self.eps_end, self.eps_decay * self.eps)

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step != 0:
            return

        if len(self.memory) < self.batch_size:
            return

        # If enough samples are available in memory, get random subset and learn
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        # Note: we multiply by 1-dones because there are no next states for terminal states!

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # Optimize the loss
        self.optimizer.zero_grad()  # Zero the accumulated gradients (pytorch accumulates gradients after each backward)
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.softly_update_target_model()

    def softly_update_target_model(self):
        '''
        Softly update the target model (network)
        '''
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def get_agent_model(self):
        return self.qnetwork_local


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)