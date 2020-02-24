from collections import namedtuple, deque
import numpy as np
import torch

def trainer(env, agent, num_episodes=100000, num_steps_in_episode=5000, render=True,
            solved_checkpoint_name='checkpoint.pth'):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for episode in range(1, num_episodes + 1):
        state, episode_score = env.reset(), 0
        for t in range(1, num_steps_in_episode + 1):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if render:
                env.render()
            agent.learn(state=state, action=action, reward=reward, next_state=next_state, done=done,
                        last_step_in_episode=(t == num_steps_in_episode))
            state = next_state
            episode_score += reward
            if done:
                break

        scores_window.append(episode_score)  # save most recent score
        scores.append(episode_score)  # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tBest Score: {:.2f}'.format(episode, np.amax(scores_window)))
        if np.mean(scores_window) >= 475:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.get_agent_model().state_dict(), solved_checkpoint_name)
            break
    return scores