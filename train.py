import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from torch.optim import Adam
import gymnasium as gym

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor
from tianshou.utils.net.discrete import Critic as CriticDiscrete
from tianshou.utils.net.continuous import ActorProb
from tianshou.utils.net.continuous import Critic as CriticContinuous

from SimRobotAEC import getSimRobotEnv


class ActorNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.net = Net(
            state_shape,
            hidden_sizes=[64, 64],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.action_head = nn.Linear(64, action_shape)

    def forward(self, obs, state=None, info={}):
        logits = self.action_head(self.net(obs))
        return logits, state


class CriticNetwork(nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.net = Net(
            state_shape,
            hidden_sizes=[64, 64],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, state=None, info={}):
        values = self.value_head(self.net(obs))
        return values, state


def get_agents(env):
    agents = []
    for agent_id in env.agents:
        obs_space = env.observation_space
        action_space = env.action_space

        if isinstance(obs_space, gym.spaces.Dict):
            # Handle dictionary observation space
            obs_shape = sum(
                space.shape[0] if len(space.shape) > 0 else 1
                for space in obs_space.values()
            )
        elif isinstance(obs_space, gym.spaces.Box):
            obs_shape = obs_space.shape
        else:
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

        if isinstance(action_space, gym.spaces.Discrete):
            action_shape = action_space.n
            net = Net(
                obs_shape,
                hidden_sizes=[64, 64],
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            actor = Actor(
                net, action_shape, device="cuda" if torch.cuda.is_available() else "cpu"
            )
            critic = CriticDiscrete(
                net, device="cuda" if torch.cuda.is_available() else "cpu"
            )
            dist_fn = lambda logits: Categorical(logits=logits)
        elif isinstance(action_space, gym.spaces.Box):
            action_shape = action_space.shape[0]
            net = Net(
                obs_shape,
                hidden_sizes=[64, 64],
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            actor = ActorProb(
                net,
                action_shape,
                unbounded=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            critic = CriticContinuous(
                net, device="cuda" if torch.cuda.is_available() else "cpu"
            )
            dist_fn = lambda x: Normal(x[0], x[1])
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

        optim = Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-4)

        policy = PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            action_space=action_space,
            eps_clip=0.2,
            value_clip=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            reward_normalization=True,
            action_scaling=True,
            action_bound_method="clip",
        )
        agents.append(policy)

    return MultiAgentPolicyManager(policies=agents, env=env)


if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment
    env = getSimRobotEnv(render_mode=None)  # Set to "human" for visualization
    env = PettingZooEnv(env)

    # Step 2: Create policies for each agent
    policies = get_agents(env)

    # Step 3: Convert the env to vector format
    train_envs = DummyVectorEnv(
        [lambda: PettingZooEnv(getSimRobotEnv()) for _ in range(1)]
    )
    test_envs = DummyVectorEnv(
        [lambda: PettingZooEnv(getSimRobotEnv()) for _ in range(1)]
    )

    # Step 4: Setup collectors
    train_collector = Collector(
        policies,
        train_envs,
        VectorReplayBuffer(20000, len(train_envs)),
    )
    test_collector = Collector(policies, test_envs)

    # Step 5: Training
    result = OnpolicyTrainer(
        policy=policies,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=1000,
        repeat_per_collect=10,
        episode_per_test=10,
        batch_size=256,
        step_per_collect=2000,
        stop_fn=lambda mean_reward: mean_reward >= 195,
    )
    print(f'Finished training! Use {result["duration"]}')

    # Step 6: Save the trained policy
    torch.save(policies.state_dict(), "robot_policies.pth")

    # Optional: Visualize a trained episode
    # env = PettingZooEnv(getSimRobotEnv(render_mode="human"))
    # collector = Collector(policies, env)
    # collector.collect(n_episode=1, render=0.1)
