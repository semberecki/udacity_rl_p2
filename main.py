from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

from ddpg_loop import ddpg
from ddpg_agent import Agent
import torch


def print_demo(env, action_size):
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    num_agents = len(env_info.agents)

    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


def setup(multi_env):
    if multi_env:
        env = UnityEnvironment(file_name='./Reacher_Linux20/Reacher.x86_64')
    else:
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
    states = env_info.vector_observations # get the current state (for each agent)
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    return env, state_size, action_size

def plot_results(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    run_demo=False
    multi_env=True
    eval_only=False

    env, state_size, action_size = setup(multi_env)
    if run_demo:
        print_demo(env, action_size)

    if eval_only:
        actor_path='checkpoints/ddpg/115_checkpoint_actor_solved.pth'
        critic_path='checkpoints/ddpg/115_checkpoint_critic_solved.pth'

        agent_eval = Agent(state_size=state_size, action_size=action_size, random_seed=0)
        agent_eval.actor_local.load_state_dict(torch.load(actor_path))
        agent_eval.critic_local.load_state_dict(torch.load(critic_path))
        scores_eval_checkpoint = ddpg(env, agent_eval,train_mode=True,
                              update_network=False,
                              n_episodes=100, score_list_len=100)

        plot_results(scores_eval_checkpoint)

    else:
        agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)
        scores = ddpg(env, agent, n_episodes=1000, checkpoints_dir="checkpoints/ddpg")
        plot_results(scores)


    env.close()