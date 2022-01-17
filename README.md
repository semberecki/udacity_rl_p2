# udacity_rl_p2 - Reacher 20

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
The environment is considered to be solved if agent gets 30 points for average of 20 agents in single episode.
Also the average score is calculated over 100 consecutive episodes.

The solution is described in ```report.md``` file

## Getting Started

Install Python 3.6 and then
```
cd python && pip install -e . 
unzip Reacher_Linux20.zip
python -m ipykernel install --user --name drlnd --display-name "drlnd"
jupyter notebook
```

## Project sructure

1. ```main.py``` contains the learning algorithm procedure.
2. ```model.py``` requires the network
3. ```ddpg_agent``` contains the DQN algorithm
4. ```checkpoints``` directory with saved checkpoints.
5. ```images``` contains graphics.

