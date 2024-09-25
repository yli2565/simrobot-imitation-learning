from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy

# Not finished yet !!!!!
from SimRobotAEC import getSimRobotEnv

if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment
    env = getSimRobotEnv(render_mode="human")

    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)
    # No idea why reset() is called in PettingZooEnv's __init__

    # Step 3: Define policies for each agent
    policies = MultiAgentPolicyManager(
        policies=[
            RandomPolicy(action_space=env.action_space),
            RandomPolicy(action_space=env.action_space),
        ],
        env=env,
    )

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=1, render=0.1, reset_before_collect=True)
