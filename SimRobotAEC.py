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
from TypedShmem import ShmemHeap, ShmemAccessor, SDict, SList
from Utils import (
    GameState,
    State,
    GameControllerState,
    ObservationJosh,
    cfg2dict,
    kill_process,
    is_zombie,
    generateRos2,
    generateSceneCon,
    generateLogCon,
    RobotSelector,
    generatePoses,
    opponentPenaltyArea,
    opponentGoalArea,
)

# these paths must match the ones used by the c++ code
# BADGER_RL_SYSTEM_DIR = Path("/root/autodl-tmp/BadgerRLSystem/")
BADGER_RL_SYSTEM_DIR = Path("/home/yuhao2024/Documents/SimRobotAEC/BadgerRLSystem/")
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
    env = SimRobotEnv(render_mode=render_mode)
    # This wrapper is only for environments which print results to the terminal
    # if render_mode == "ansi":
    #     env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
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

        # PettingZoo required fields
        self.possible_agents = self.agentRobots

    # Helper fields
    @property
    def team1(self):
        return list(filter(lambda x: 0 < x < 20, self.agentRobots))

    @property
    def team2(self):
        return list(filter(lambda x: 20 < x < 40, self.agentRobots))

    @property
    def teams(self):
        return {0: self.team1, 1: self.team2}

    @property
    def moveableAgents(self):
        return [*self.agentRobots, *self.BhumanRobots, *self.hijackedRobots]

    @property
    def allRobots(self):
        return [
            *self.agentRobots,
            *self.dummyRobots,
            *self.BhumanRobots,
            *self.hijackedRobots,
        ]

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

        # Scene
        self.sceneName = "PythonEnvGenerated"
        self.logConFileName = "logCon"
        self.team1Number = 5
        self.team2Number = 70

        # Simulator pid
        self.simulator_pid = 0  # uninitialized

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

        RobotTacticList = [0] * 40
        for robot in self.agentRobots:
            RobotTacticList[robot] = RobotTactic.POLICY_CONTROL.value

        for robot in self.dummyRobots:
            RobotTacticList[robot] = RobotTactic.DUMMY.value

        for robot in self.BhumanRobots:
            RobotTacticList[robot] = RobotTactic.BHUMAN_CONTROL.value

        for robot in self.hijackedRobots:
            RobotTacticList[robot] = RobotTactic.STATIC_CONTROL.value

        self.environmentVariables = {}
        for robot in self.possible_agents:
            self.environmentVariables[robot] = {
                "PythonEnvPrefix": str(os.getpid()) + "_" if not DEBUG else "DEBUG_",
                "RobotTacticList": RobotTacticList,
                "RobotStateFlagArrayShmSize": len(self.agentRobots),
                "ActionArrayShmSize": int(np.prod(self.action_space(robot).shape)),
                "ObsArrayShmSize": int(np.prod(self.observation_space(robot).shape)),
                "GroundTruthArrayShmSize": GROUND_TRUTH_SIZE,
                "InitialPoseArrayShmSize": 6 * len(self.agentRobots),
                "UpdatePeriod": 500,
                "Team1Number": self.team1Number,
                "Team2Number": self.team2Number,
                "CalibrationPeriod": 100000,
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

    def writeScenes(self):
        logConName = self.sceneName + "_LogConfiguration"
        # Generate .ros2 file and .con file
        self.scene = {
            "ros2": generateRos2(
                robots=self.moveableAgents,
                dummyRobots=self.dummyRobots,
                team1Number=self.team1Number,
                team2Number=self.team2Number,
            ),
            "scene_con": generateSceneCon(logConName),
            "log_con": generateLogCon(),
        }
        # Write the ros2 and con file
        with open(
            BADGER_RL_SYSTEM_DIR / f"Config/Scenes/{self.sceneName}.ros2", "w"
        ) as ros2File:
            ros2File.write(self.scene["ros2"])
        with open(
            BADGER_RL_SYSTEM_DIR / f"Config/Scenes/{self.sceneName}.con", "w"
        ) as conFile:
            conFile.write(self.scene["scene_con"])
        with open(
            BADGER_RL_SYSTEM_DIR / f"Config/Scenes/Includes/{logConName}.con", "w"
        ) as conFile:
            conFile.write(self.scene["log_con"])

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

            env = {**env, **self.environmentVariables[self.possible_agents]}

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
        if self.simulator_pid == 0:
            self.writeScenes()
            if DEBUG:
                self.simulator_pid = -1  # launch the simulator manually
            elif self.simulator_pid == -1:
                self.startSimRobot()

        if self.simulator_pid != -1 and is_zombie(self.simulator_pid):
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
        while not GameControllerState.STATE_READY == GameControllerState(
            self.gameStateFlagShm[0]
        ):
            pass

        # Calculate and send the reset pos
        initialRobotPoses: List[Point] = generatePoses(
            opponentPenaltyArea, len(self.allRobots), seed=seed
        )
        initialBallPose: Point = generatePoses(opponentGoalArea, 1, seed=seed)[0]

        # Build the initial pose array
        purposedInitialPose = {"Ball": [initialBallPose.x, initialBallPose.y]}
        for idx, robot in enumerate(self.allRobots):
            robotPos = initialRobotPoses[idx]
            rot = math.atan2(
                initialBallPose.y - robotPos.y, initialBallPose.x - robotPos.x
            )
            purposedInitialPose[str(robot)] = [
                robotPos.x,
                robotPos.y,
                350.0,
                0.0,
                0.0,
                rot,
            ]

        self.initialPoses.set(purposedInitialPose)
        self.initialPosesShm.postCounterSem()

        # Wait until game state is "set"
        while not GameControllerState.STATE_SET == GameControllerState(
            self.gameStateFlagShm[0]
        ):
            pass

        self._agent_selector = RobotSelector(
            self.possible_agents,
            {
                self.possible_agents[idx]: self.robotShmManagers[agent]["ObsArrayShm"]
                for idx, agent in enumerate(self.agentRobots)
            },
        )

        # This will wait until a robot return the first observation
        self.agent_selection = self._agent_selector.next()
        print(f"Agent selection: {self.agent_selection}")

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
        print(f"Agent selection: {self.agent_selection}")

    def openShm(self):
        if DEBUG:
            pythonEnvPrefix = "DEBUG_"
        else:
            pythonEnvPID = os.getpid()
            pythonEnvPrefix = str(pythonEnvPID) + "_"

        # Config shm (Switch to new TypedShmem communication module)
        self.globalConfigShm = ShmemHeap(pythonEnvPrefix + "GlobalConfig")
        self.globalConfigShm.create()
        self.globalConfig = ShmemAccessor(self.globalConfigShm)

        self.initialPosesShm = ShmemHeap(pythonEnvPrefix + "InitialPoses")
        self.initialPosesShm.create()
        self.initialPoses = ShmemAccessor(self.initialPosesShm)

        self.globalConfig.set(self.environmentVariables[self.possible_agents[0]])

        # Global shm
        self.globalShmManager = SharedMemoryManager(
            str(pythonEnvPID) if not DEBUG else "DEBUG",
            [
                (
                    "ActionUpdatedFlagArrayShm",
                    (len(self.agentRobots),),
                    SHM_VERBOSE,
                ),
                ("RobotStateFlagArrayShm", (len(self.agentRobots),), SHM_VERBOSE),
                ("GameStateFlagShm", (1,), SHM_VERBOSE),
            ],
        )
        # Alias of some global shm
        self.actionUpdatedFlagArrayShm = self.globalShmManager[
            "ActionUpdatedFlagArrayShm"
        ]
        self.robotStateFlagArrayShm = self.globalShmManager["RobotStateFlagArrayShm"]
        self.gameStateFlagShm = self.globalShmManager["GameStateFlagShm"]

        self.globalShmManager.createTunnels()

        self.actionUpdatedFlagArrayShm.sendArray([0] * len(self.possible_agents))

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
