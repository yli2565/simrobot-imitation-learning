# AbstractSim

This repository contains the BadgerRL lab's abstract robosoccer environment simulators, as well as reference policies trained using these simulators. You can configure the main function of environments to either train a new policy, or load and demonstrate rollouts with an existing policy. 

### run.py

This allows the policies to be trained or rendered.

Nagivate to the abstractSim directory.

To train a policy:
````
python3 run.py --train --env=<name of environment>

for example:
python3 run.py --train --env=kick_to_goal

see below for more arguments
````

To render a policy:
```
python3 run.py --render --env=<name of environment> --policy_path=<path to trained policy>

for example:

python3 run.py --render --env=kick_to_goal --policy_path=./policies/kick_to_goal_policy.zip
```

## environment
````
python3.9 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
````

It is recommended to make a pythonvirtual env using requirements.txt before using the code in this repository, this will ensure you can load the saved policies, and ensure the policies you train are compatible with other people's environments.  You will need to use python 3.9.

# Multi-agent vs Single-agent
While there is only one folder, abstractSim, the environments in that folder are based on pettingzoo which allows you to change the number of agents in an environment. If you navigate to kick_to_goal.py under envs, you can see an example.

# Example environments
There should be at least one example environment already created.

`./abstractSim/multi_kick_ball_to_goal.py` 

It trains policies that get the ball to the goal. To iterate and create your own environments it is recommended that you create a copy of the environment and edit it from there. That way you are able to initially figure out the structure.

Note: This environment is NOT perfect by any means. It was put together for example purposes and as starter code. Feel free to change anything in a copy of it!!

### export.py

export.py can be used to export a stable-baselines3 saved policy into a format that can be read by the C++ code base. 

For example, the following will create `abstractSim/exported/policy.onnx`:
```
cd abstractSim
python export.py policies/kick_to_goal_policy.zip
```

When export.py is executed, it will create a policy.onnx file. You need to copy that into Config/policies/<environment name> in the C++ repository to try them out in the c++ environments.


## Other useful arguments
**Flags**

--train -> use to set in training mode

--env=<NAME_OF_ENV> -> use to set name of ENV to train. NOTE: Use NAME not PATH 
  
--total_timesteps=<x> -> Set number of training timesteps (Default 10 million)
  
--batch_size=<x> -> Set batch size (Default 64)
  
--vec_envs=<x> -> Set number of SB3 vectorized envs (Default 12)

--wandb -> Flag to use weights and biases to track. Must have wandb initialized to use
  