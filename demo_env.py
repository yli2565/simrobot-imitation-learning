import SimRobotAEC

if __name__ == "__main__":
    env = SimRobotAEC.getSimRobotEnv(render_mode="human")
    env.reset(seed=55)

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
