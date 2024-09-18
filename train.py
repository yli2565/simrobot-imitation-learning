import SimRobotAEC
from SimRobotAEC import SimRobotEnv

env = SimRobotAEC.env(render_mode="human")
env.reset(seed=55)

while True:
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(env.agent_selection).sample()

    env.step(action)
env.close()
