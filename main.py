import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


from unityagents import UnityEnvironment
import numpy as np

from ddpg_agent import Agent


def ddpg(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, train_mode=True):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf

    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state =  env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            if train_mode:
                agent.step(state, action, reward, next_state, done)
            score += reward

            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    return scores



if __name__ == '__main__':

    env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    # scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    # while True:
    #     actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    #     actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    #     env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    #     next_states = env_info.vector_observations         # get next state (for each agent)
    #     rewards = env_info.rewards                         # get reward (for each agent)
    #     dones = env_info.local_done                        # see if episode finished
    #     scores += env_info.rewards                         # update the score (for each agent)
    #     states = next_states                               # roll over states to next time step
    #     if np.any(dones):                                  # exit loop if episode finished
    #         break
    # print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)
    scores = ddpg(env, agent, n_episodes=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


    env.close()
