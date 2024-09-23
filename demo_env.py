import SimRobotAEC
from SimRobotAEC import SimRobotEnv

env = SimRobotAEC.getSimRobotEnv(render_mode="human")
env.reset(seed=55)
# env.observe(env.agent_selection)
# env.step(env.action_space(env.agent_selection).sample())
while True:
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(env.agent_selection).sample()

    env.step(action)

    if all(env.terminations.values()):
        env.reset()
env.close()
