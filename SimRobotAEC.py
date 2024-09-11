import functools
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List
from enum import Enum
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from shapely import Point

from InterThreadCommunication import SharedMemoryHelper, SharedMemoryManager
from Utils import (
    GameState,
    State,
    ObservationJosh,
    cfg2dict,
    kill_process,
    is_zombie,
    generateRos2,
    generateCon,
    RobotSelector,
    generatePoses,
    opponentPenaltyArea,
    opponentGoalArea,
)

# these paths must match the ones used by the c++ code
# BADGER_RL_SYSTEM_DIR = Path("/root/autodl-tmp/BadgerRLSystem/")
BADGER_RL_SYSTEM_DIR = Path("/home/yuhao2024/Documents/BadgerRLSystem-josh/")
# BADGER_RL_SYSTEM_DIR = Path("/home/chkxwlyh/Documents/Study/RL100/BadgerRLSystem/")

DEBUG_PRINTS = False
DEBUG = True

RAW_OBS_SIZE = 100
OBS_SIZE = 5
RAW_ACT_SIZE = 20
ACT_SIZE = 3
INFO_SIZE = 55
GROUND_TRUTH_SIZE = 5
EPISODE_TIME_LIMIT = 60 * 1000  # (ms)
SHM_VERBOSE = 2
CALC_OBS_IN_PYTHON = False
CALC_ACT_IN_PYTHON = False
CALC_REWARD_IN_PYTHON = True


class RobotTactic(Enum):
    NONE = 0
    BHUMAN_CONTROL = 1
    POLICY_CONTROL = 2
    STATIC_CONTROL = 3
    DUMMY = 4  # play dead


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = SimRobotEnv(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    # if render_mode == "ansi":
    #     env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class SimRobotEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "SimRobotTraining_v0"}

    def initialize_agents(self):
        # agents
        # Team 1's robot is in [0, 19], Team 2's robot is in [20, 39]
        # The agent number of a robot is its robot number - 1. For example, robot 1 is agent 0

        # Each robot is controlled by a RL agent
        self.agentRobots = [4, 5, 24]  # Use PolicyControl
        # Each dummy robot is just an obstacle. Their position can be set during reset
        self.dummyRobots = [1, 21]  # Play Dead
        self.BhumanRobots = (
            []
        )  # TODO: Implement robots following Bhuman code # Use BHuman's control logic
        self.hijackedRobots = (
            []
        )  # TODO: Implement robots following hard coded logic # Use Static Control

        set1, set2, set3, set4 = (
            set(self.agentRobots),
            set(self.dummyRobots),
            set(self.BhumanRobots),
            set(self.hijackedRobots),
        )
        if (
            (set1 & set2)
            or (set1 & set3)
            or (set1 & set4)
            or (set2 & set3)
            or (set2 & set4)
            or (set3 & set4)
        ):
            raise ValueError("Robot cannot be in multiple teams at the same time.")

        self.possible_agents = {"Robot{}".format(rn) for rn in self.agentRobots}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Helper fields
        self.team1 = list(filter(lambda x: 0 < x < 20, self.agentRobots))
        self.team2 = list(filter(lambda x: 20 < x < 40, self.agentRobots))
        self.teams = {0: self.team1, 1: self.team2}

        # Robot that can move
        moveableAgents = [*self.agentRobots, *self.BhumanRobots, *self.hijackedRobots]

        # Scene
        self.sceneName = "PythonEnvGenerated"
        # Generate .ros2 file and .con file
        self.scene = {
            "ros2": generateRos2(moveableAgents, self.dummyRobots),
            "con": generateCon(),
        }

        RobotTacticList = [0] * 40
        for robot in self.agentRobots:
            RobotTacticList[robot] = RobotTactic.POLICY_CONTROL

        for robot in self.dummyRobots:
            RobotTacticList[robot] = RobotTactic.DUMMY

        for robot in self.BhumanRobots:
            RobotTacticList[robot] = RobotTactic.BHUMAN_CONTROL

        for robot in self.hijackedRobots:
            RobotTacticList[robot] = RobotTactic.STATIC_CONTROL

        self.environmentVariables = {}
        for robot in self.agentRobots:
            self.environmentVariables[robot] = {
                "PythonEnvPrefix": str(os.getpid()) + "_" if not DEBUG else "",
                "RobotTacticList": RobotTacticList,
                "RobotStateFlagArrayShmSize": len(self.agentRobots),
                "ActionArrayShmSize": int(
                    np.prod(self.action_space(self.possible_agents[robot]).shape)
                ),
                "ObsArrayShmSize": int(
                    np.prod(self.observation_space(self.possible_agents[robot]).shape)
                ),
                "GroundTruthArrayShmSize": GROUND_TRUTH_SIZE,
                "InitialPoseArrayShmSize": 6 * len(self.agentRobots),
                "UpdatePeriod": 0.5,
                "CalibrationPeriod": 100000,
            }

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.initialize_agents()

        self._action_spaces = {
            agent: Box(
                np.array([-1] * ACT_SIZE),
                np.array([1] * ACT_SIZE),
            )
            for agent in self.possible_agents
        }

        self._observation_spaces = {
            agent: Box(low=-1, high=1, shape=(OBS_SIZE,))
            for agent in self.possible_agents
        }

        self.render_mode = render_mode

        self.openShm()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        gymnasium.logger.warn(
            "You are calling render method without specifying any render mode."
        )

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def startSimRobot(self):
        if sys.platform.startswith("win"):
            raise NotImplementedError("Launching SimRobot on Windows not supported")
        elif sys.platform.startswith("darwin"):
            # MACOS specific command
            command = [
                "open",
                "-g",
                BADGER_RL_SYSTEM_DIR / "Config/Scenes/randomScene.ros2",
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=BADGER_RL_SYSTEM_DIR,
            )
        elif sys.platform.startswith("linux"):
            # Compile the SimRobot binary
            compileCommand = [
                BADGER_RL_SYSTEM_DIR / "Make/Linux/compile",
                "Release",
                "SimRobot",
            ]
            compileProcess = subprocess.Popen(
                compileCommand,
                stdout=subprocess.DEVNULL,
                cwd=BADGER_RL_SYSTEM_DIR,
            )
            compileProcess.wait()

            # Launch the SimRobot Simulator
            runCommand = [
                str(BADGER_RL_SYSTEM_DIR / "Build/Linux/SimRobot/Release/SimRobot"),
                "-g",
                str(BADGER_RL_SYSTEM_DIR / f"Config/Scenes/{self.sceneName}.ros2"),
            ]

            env = os.environ.copy()

            env = {**env, **self.environmentVariables[self.possible_agents[0]]}

            # Write the ros2 and con file
            with open(
                BADGER_RL_SYSTEM_DIR / f"Config/Scenes/{self.sceneName}.ros2", "w"
            ) as ros2File:
                ros2File.write(self.scene["ros"])
            with open(
                BADGER_RL_SYSTEM_DIR / f"Config/Scenes/{self.sceneName}.con", "w"
            ) as conFile:
                conFile.write(self.scene["con"])

            # TODO: change the output and error file name
            with open("output.txt", "w") as outFile, open("error.txt", "w") as errFile:
                process = subprocess.Popen(
                    runCommand,
                    stdout=outFile,
                    stderr=errFile,
                    cwd=BADGER_RL_SYSTEM_DIR,
                    env=env,
                )

        else:
            raise NotImplementedError("Unsupported platform")

        self.simulator_pid = process.pid
        return

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        # Reset simulator (If the simulator is not running, launch it)
        if DEBUG:
            self.simulator_pid = -1  # launch the simulator manually
        elif self.simulator_pid == -1:
            self.startSimRobot()
        elif self.simulator_pid != -1 and is_zombie(self.simulator_pid):
            # Restart the simulator
            kill_process(self.simulator_pid)
            self.startSimRobot()

        # Reset data
        self.agents = self.possible_agents[:]
        self.groundTruths = {agent: None for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        # Wait until game state is "ready"
        while not GameState.isReady(State(self.gameStateFlagShm[0])):
            pass

        # Calculate and send the reset pos
        initialRobotPoses: List[Point] = generatePoses(
            opponentPenaltyArea, len(self.agentRobots), seed=0
        )
        initialBasePose: Point = generatePoses(opponentGoalArea, 1, seed=0)[0]

        # Build the initial pose array
        initialPosArray = []
        for robotPos in initialRobotPoses:
            rot = math.atan2(
                initialBasePose.y - robotPos.y, initialBasePose.x - robotPos.x
            )
            initialPosArray += [robotPos.x, robotPos.y, 350.0, 0, 0, rot]

        self.initialPoseArrayShm.sendArray(initialPosArray)

        # Wait until game state is "set"
        while not GameState.isSet(State(self.gameStateFlagShm[0])):
            pass

        self._agent_selector = RobotSelector(
            self.possible_agents,
            {
                agent: self.robotShmManagers[agent]["ObsArrayShm"]
                for agent in self.possible_agents
            },
        )

        # This will wait until a robot return the first observation
        self.agent_selection = self._agent_selector.next()

    def calcReward(self, groundTruth):
        """
        Calculate the reward for the agent from the ground truth information
        """
        return 1

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # Fetch the corresponding observation array from shared memory
        observation = self.robotShmManagers[agent]["ObsArrayShm"].fetchArray()
        self.observations[agent] = observation
        # Fetch ground truth / reward from SimRobot and set robot's reward
        groundTruth = self.robotShmManagers[agent]["GroundTruthArrayShm"].fetchArray()
        self.groundTruths[agent] = groundTruth

        return np.array(observation)

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """

        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # Calculate reward
        groundTruth = self.groundTruths[agent]
        self.rewards[agent] = self.calcReward(groundTruth)

        # Update termination / truncation according to ground truth
        # TODO: make this correct
        groundTruthGameState = State(groundTruth[0])
        if not GameState.isPlaying(groundTruthGameState):
            self.terminations[self.agent_selection] = True

        # If a robot is penalized, truncation is true
        if GameState.isPenalized(groundTruthGameState):
            self.truncations[self.agent_selection] = True

        # Perform new action
        agentActionShm = self.robotShmManagers[agent]["ActionArrayShm"]
        agentActionShm.sendArray(action)
        while agentActionShm.probeSem() != 0:
            pass  # Wait for action to be received by simulated robot

        if self._agent_selector.is_last():
            self.num_moves += 1

            # CONFUSED: Why observe the current state here ?
            # observe the current state
            # for i in self.agents:
            #     self.observations[i] = self.state[
            #         self.agents[1 - self.agent_name_mapping[i]]
            #     ]

        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
        self._clear_rewards()

        # TODO: enable simple rendering
        # if self.render_mode == "human":
        #     self.render()

        # selects the next agent.
        # 99% of the time, we will wait here
        self.agent_selection = self._agent_selector.next()

    def openShm(self):
        if DEBUG:
            pythonEnvPrefix = ""
        else:
            pythonEnvPID = os.getpid()
            pythonEnvPrefix = str(pythonEnvPID) + "_"

        # Global shm
        self.globalShmManager = SharedMemoryManager(
            str(pythonEnvPID) if not DEBUG else "",
            [
                ("ActionUpdatedFlagArrayShm", (1,), SHM_VERBOSE),
                ("RobotStateFlagArrayShm", (len(self.agentRobots),), SHM_VERBOSE),
                ("GameStateFlagShm", (1,), SHM_VERBOSE),
                ("InitialPoseArrayShm", (len(self.agentRobots) * 6,), SHM_VERBOSE),
            ],
            pythonEnvPrefix == "",
        )
        # Alias of some global shm
        self.actionUpdatedFlagArrayShm = self.globalShmManager[
            "ActionUpdatedFlagArrayShm"
        ]
        self.robotStateFlagArrayShm = self.globalShmManager["RobotStateFlagArrayShm"]
        self.gameStateFlagShm = self.globalShmManager["GameStateFlagShm"]
        self.initialPoseArrayShm = self.globalShmManager["InitialPoseArrayShm"]

        self.globalShmManager.createTunnels()

        # Robot individual shm
        self.robotShmManagers: Dict[int, SharedMemoryManager] = {
            robot: SharedMemoryManager(
                pythonEnvPrefix + "robot" + str(robot),
                [
                    ("ObsArrayShm", (OBS_SIZE,), SHM_VERBOSE),
                    ("ActionArrayShm", (ACT_SIZE,), SHM_VERBOSE),
                    ("GroundTruthArrayShm", (GROUND_TRUTH_SIZE,), SHM_VERBOSE),
                ],
            )
            for robot in self.agentRobots
        }

        for robot in self.agentRobots:
            self.robotShmManagers[robot].createTunnels()

    def closeShm(self):
        self.globalShmManager.close()
        self.globalShmManager.unlink()
        for robot in self.robots:
            self.robotShmManagers[robot].close()
            self.robotShmManagers[robot].unlink()

    # def clearShm(self):
