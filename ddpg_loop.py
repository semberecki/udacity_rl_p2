from collections import deque

import numpy as np
import torch


def ddpg(env, agent, n_episodes=3000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.999,
         train_mode=True, update_network=True, score_list_len=100, checkpoints_dir="checkpoints"):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores = []                        # list containing scores from each episode
    scores_deque = deque(maxlen=score_list_len)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    brain_name = env.brain_names[0]
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            if not update_network:
                eps = 0.0
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            if update_network:
                agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        if update_network:
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % score_list_len == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        if i_episode % 200 == 0 and update_network:
            torch.save(agent.actor_local.state_dict(), f"{checkpoints_dir}/{i_episode}_checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), f"{checkpoints_dir}/{i_episode}_checkpoint_critic.pth")

    return scores