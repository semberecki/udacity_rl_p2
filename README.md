# udacity_rl_p2 - Reacher

## Project Details


### State space

```

INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		goal_speed -> 1.0
		goal_size -> 5.0
Unity brain name: ReacherBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 33
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , ,   
```
The environment is considered to be solved if agent gets 30 points for average of 1 or 20 agents in single episode.
Also the average score is calculated over 100 consecutive episodes.

```
Number of agents: 20
Size of each action: 4
There are 20 agents. Each observes a state with length: 33
```

```
Number of agents: 1
Size of each action: 4
There are 1 agents. Each observes a state with length: 33
```

The algorithm was solved both for the Reacher 20 agents and Reacher 1 agent environment.



The solution is described in ```report.md``` file

## Getting Started

Install Python 3.6 and then
```
cd python && pip install -e . 
unzip Reacher_Linux20.zip
unzip Reacher_Linux.zip
python -m ipykernel install --user --name drlnd --display-name "drlnd"
jupyter notebook
```

## Project sructure

1. ```ddpg_agent.py``` - contains basic DDPG implementation 
2. ```ddpg_loop.py``` - defines workflow of algorithm
3. ```main.py``` - headless run
4. ```model.py``` - defines used networks
5. ```Reacher-Solution-ddpg.ipynb``` - solution with results in jupyter
6. ```Reacher-Solution-ddpg.html``` - solution with results - html
7. ```Reacher_Linux20.zip``` - binaries version 20 agents
8. ```Reacher_Linux.zip``` - binaries version 1 agent
9. ```README.md``` contains the DQN algorithm
10. ```report.md``` directory with saved checkpoints.
11. ```images``` contains graphics.
12. ``` python ``` contains Unity Agents and other required packages

