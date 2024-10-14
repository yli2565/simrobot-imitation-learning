import functools
import math
import os
import random
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import gymnasium
import numpy as np
from gymnasium.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from shapely import Point
from TypedShmem import SDict, ShmemAccessor, ShmemHeap, SList

from InterThreadCommunication import SharedMemoryHelper, SharedMemoryManager
from Utils import (
    GameControllerState,
    GameState,
    RobotSelector,
    SoccerFieldAreas,
    SoccerFieldPoints,
    State,
    generateLogCon,
    generatePoses,
    generateRos2,
    generateSceneCon,
    is_zombie,
    kill_process,
    should_use_vglrun,
)

WISTEX_SYSTEM_DIR = Path(__file__).parent / "wistex-system"

DEBUG_PRINTS = False
DEBUG = False

RAW_OBS_SIZE = 100
OBS_SIZE = 5
RAW_ACT_SIZE = 20
ACT_SIZE = 3
INFO_SIZE = 55
GROUND_TRUTH_SIZE = 5
EPISODE_TIME_LIMIT = 60 * 1000  # (ms)
SHM_VERBOSE = 2


class RobotTactic(Enum):
    """The tactic of a robot"""

    NONE = 0  # Not used
    BHUMAN_CONTROL = 1  # play with Bhuman policy
    POLICY_CONTROL = 2  # play with RL policy (communicate with python)
    STATIC_CONTROL = 3  # play static logic
    DUMMY = 4  # play dead


def getSimRobotEnv(render_mode=None, **kwargs):
    """Create a SimRobot env with layers of wrappers for exception handling."""
    env = SimRobotEnv(render_mode=render_mode, **kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class SimRobotEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "SimRobotTraining_v0"}

    def __init__(
        self,
        render_mode=None,
        **kwargs,
    ):
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

        self.initializeAgents()

        # Scene
        self.sceneName = "PythonEnvGenerated"
        self.logConFileName = "logCon"
        self.team1Number = 5
        self.team2Number = 70

        # Simulator pid
        self.simulator_pid = 0  # uninitialized
        self.env_id = kwargs.get("env_id", random.randint(0, 100000))

        self.pythonEnvPid = os.getpid()
        self.pythonEnvPrefix = (
            f"{self.pythonEnvPid}_{self.env_id}_" if not DEBUG else "DEBUG_"
        )

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
                "PythonEnvPrefix": self.pythonEnvPrefix,
                "RobotTacticList": RobotTacticList,
                "RobotStateFlagArrayShmSize": len(self.agentRobots),
                "ActionArrayShmSize": int(np.prod(self.action_space(robot).shape)),
                "ObsArrayShmSize": int(np.prod(self.observation_space(robot).shape)),
                "InitialPoseArrayShmSize": 6 * len(self.agentRobots),
                "UpdatePeriod": 500,
                "Team1Number": self.team1Number,
                "Team2Number": self.team2Number,
                "CalibrationPeriod": 100000,
            }

        self.render_mode = render_mode

        self.configShm()
        self.openShm()

        def interruptCallback():
            if GameControllerState.STATE_READY == GameControllerState(
                self.gameStateFlagShm[0]
            ):
                return True
            return False

        self.interruptCallback = interruptCallback
        self.rng = np.random.default_rng(55)

    def __del__(self):
        self.close()

    # Gym API Required

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
        print("Current Simulator PID:", self.simulator_pid)
        if self.simulator_pid == 0:
            self.writeScenes()
            self.simulator_pid = self.launchSimulator(DEBUG)
            print("Simulator launched.")
        elif self.simulator_pid > 0:
            # Relaunch the simulator if it is dead for some reason
            if not self.checkSimulatorStatus():
                self.closeSimulator()
                self.simulator_pid = self.launchSimulator(DEBUG)

        # Handle duplicated reset
        if hasattr(self, "num_moves") and self.num_moves == 0:
            return

        # Reset data
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.extendedInfos = {agent: None for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        # Wait until game state is "ready"
        while not GameControllerState.STATE_READY == GameControllerState(
            self.gameStateFlagShm[0]
        ):
            pass

        purposedInitialPose = self.resetBallAndRobotPositions(seed=seed)

        self.initialPoses.set(purposedInitialPose)

        # Wait until game state is "set" (all robots are move to the correct initial position)
        while not GameControllerState.STATE_SET == GameControllerState(
            self.gameStateFlagShm[0]
        ):
            if self.initialPosesShm.getCounterSemValue() == 0:
                self.initialPosesShm.postCounterSem()
            pass

        self._agent_selector = RobotSelector(
            self.agents,
            {
                self.possible_agents[idx]: self.robotShmManagers[agent][
                    "ActionRequestFlagShm"
                ]
                for idx, agent in enumerate(self.agentRobots)
            },
            self.interruptCallback,
        )

        # This will wait until a robot return the first observation
        self.agent_selection = self._agent_selector.next()
        print(f"Reset Agent selection: {self.agent_selection}")

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # Fetch the corresponding observation array from shared memory
        agentNum = int(agent)
        obsShm = self.robotShmManagers[agentNum]["ObsArrayShm"]

        # Wait for observation when: fetching very first observation, else use old observation
        if obsShm.probeSem() != 0 or self.observations[agent] is None:
            observation = obsShm.probeArray()
            self.observations[agent] = observation

        # Fetch extended info
        while self.robotExtendedInfoShmems[agentNum].getCounterSemValue() == 0:
            pass
        self.extendedInfos[agent] = self.robotExtendedInfos[agentNum].fetch()
        print(self.extendedInfos[agent])
        return self.observations[agent]

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
            oldagent = self.agent_selection
            self._was_dead_step(action)
            print(
                "Remove agent {}, current selection {}".format(
                    oldagent, self.agent_selection
                )
            )
            return

        agent = self.agent_selection
        agentNum = int(agent)

        # Calculate reward
        extendedInfo = self.extendedInfos[agent]
        print("Robot {} ExtendedInfo: {}".format(self.agent_selection, extendedInfo))
        self.rewards[agent] = self.calcReward(extendedInfo["GroundTruth"])

        robotGameControllerStatePerception = GameControllerState(
            extendedInfo["GameControllerState"]
        )
        gameStatePerception = State(extendedInfo["GameState"])

        if GameControllerState.STATE_READY == robotGameControllerStatePerception:
            print(
                "Robot {} terminate because of {}".format(
                    self.agent_selection, "GameControllerState"
                )
            )
            self.terminations[self.agent_selection] = True

        if not GameState.isPlaying(gameStatePerception):
            print(
                "Robot {} terminate because of {}".format(
                    self.agent_selection, "RobotGameState"
                )
            )
            self.terminations[self.agent_selection] = True

        # Update termination / truncation according to ground truth
        # groundTruthGameState = State(groundTruth[0])
        # if not GameState.isPlaying(groundTruthGameState):
        #     self.truncations[self.agent_selection] = True

        # Perform new action
        agentActionShm = self.robotShmManagers[agentNum]["ActionArrayShm"]
        agentActionRequestFlagShm = self.robotShmManagers[agentNum][
            "ActionRequestFlagShm"
        ]
        agentActionShm.sendArray(action)
        agentActionRequestFlagShm.clearSem()  # Respond to the request

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
        # 99% of the time, we will wait here for the action request from the robot
        # If all robots are terminated, no need to wait
        if not all(self.terminations.values()):
            self.agent_selection = self._agent_selector.next()
            print(f"Agent selection: {self.agent_selection}")

    def _was_dead_step(self, action) -> None:
        """Minor change: Instead of using self._skip_agent_selection to determine the next agent, directly call self._agent_selector.next() to get the next agent."""

        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")

        # removes dead agent
        agent = self.agent_selection
        assert (
            self.terminations[agent] or self.truncations[agent]
        ), "an agent that was not dead as attempted to be removed"
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        # finds next dead agent or loads next live agent (Stored in _skip_agent_selection)
        _deads_order = [
            agent
            for agent in self.agents
            if (self.terminations[agent] or self.truncations[agent])
        ]
        if _deads_order:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0]
        else:
            self.agent_selection = self._agent_selector.next()
        self._clear_rewards()

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
        self.closeSimulator()

        self.closeShm()

    # Helper fields
    @property
    def team1(self):
        return list(filter(lambda x: 0 < x < 20, self.allRobots))

    @property
    def team2(self):
        return list(filter(lambda x: 20 < x < 40, self.allRobots))

    @property
    def teams(self):
        return {0: self.team1, 1: self.team2}

    @property
    def moveableRobots(self):
        return [*self.agentRobots, *self.BhumanRobots, *self.hijackedRobots]

    @property
    def allRobots(self):
        return [
            *self.agentRobots,
            *self.dummyRobots,
            *self.BhumanRobots,
            *self.hijackedRobots,
        ]

    # Core functions
    def initializeAgents(self):
        """
        Set up agents before launching the simulator
        """

        # agents
        # Team 1's robot is in [0, 19], Team 2's robot is in [20, 39]
        # The agent number of a robot is its robot number - 1. For example, robot 1 is agent 0

        # Each robot is controlled by a RL agent
        self.agentRobots = [5, 24]  # Use PolicyControl

        # Each dummy robot is just an obstacle. Their position can be set during reset
        self.dummyRobots = [1, 21]  # Play Dead
        self.BhumanRobots = [
            3,
            23,
        ]  # Robots following Bhuman code # Use BHuman's control logic
        self.hijackedRobots = [
            7,
            27,
        ]  # Robots following hard coded logic # Use Static Control logic in StaticControl.cpp

        # Verification part
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
        self.possible_agents = [str(robot) for robot in self.agentRobots]

    def writeScenes(self):
        """
        Write .ros2 and .con files with specification of the each robot's role
        """

        logConName = self.sceneName + "_LogConfiguration"
        # Generate .ros2 file and .con file
        self.scene = {
            "ros2": generateRos2(
                robots=self.moveableRobots,
                dummyRobots=self.dummyRobots,
                team1Number=self.team1Number,
                team2Number=self.team2Number,
            ),
            "scene_con": generateSceneCon(logConName),
            "log_con": generateLogCon(),
        }
        # Write the ros2 and con file
        with open(
            WISTEX_SYSTEM_DIR / f"Config/Scenes/{self.sceneName}.ros2", "w"
        ) as ros2File:
            ros2File.write(self.scene["ros2"])
        with open(
            WISTEX_SYSTEM_DIR / f"Config/Scenes/{self.sceneName}.con", "w"
        ) as conFile:
            conFile.write(self.scene["scene_con"])
        with open(
            WISTEX_SYSTEM_DIR / f"Config/Scenes/Includes/{logConName}.con", "w"
        ) as conFile:
            conFile.write(self.scene["log_con"])

    def launchSimulator(self, debug=False):
        """
        Launch the simulator based on the platform
        """
        if debug:
            return -1
        if sys.platform.startswith("win"):
            raise NotImplementedError("Launching SimRobot on Windows not supported")
        elif sys.platform.startswith("darwin"):
            # MACOS specific command
            command = [
                "open",
                "-g",
                WISTEX_SYSTEM_DIR / "Config/Scenes/randomScene.ros2",
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=WISTEX_SYSTEM_DIR,
            )
        elif sys.platform.startswith("linux"):
            # Compile the SimRobot binary
            compileCommands = [
                [WISTEX_SYSTEM_DIR / "Make/Linux/generate"],
                [
                    WISTEX_SYSTEM_DIR / "Make/Linux/compile",
                    "Release",
                    "SimRobot",
                ],
            ]
            for cmd in compileCommands:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    cwd=WISTEX_SYSTEM_DIR,
                    check=True,
                )
            # Launch the SimRobot Simulator
            runCommand = [
                str(WISTEX_SYSTEM_DIR / "Build/Linux/SimRobot/Release/SimRobot"),
                "-g",
                str(WISTEX_SYSTEM_DIR / f"Config/Scenes/{self.sceneName}.ros2"),
            ]
            env = os.environ.copy()
            env["PythonEnvPrefix"] = self.pythonEnvPrefix
            # print(" ".join(runCommand))
            # runCommand = ["taskset", "-c", "0-14"] + runCommand # Assign big cores
            if should_use_vglrun():
                runCommand = ["vglrun"] + runCommand

            with open("output.txt", "w") as outFile, open("error.txt", "w") as errFile:
                process = subprocess.Popen(
                    runCommand,
                    stdout=outFile,
                    stderr=errFile,
                    cwd=WISTEX_SYSTEM_DIR,
                    env=env,
                )
        else:
            raise NotImplementedError("Unsupported platform")
        return process.pid

    def checkSimulatorStatus(self):
        """
        Check if the simulator process is still running and not a zombie
        """
        if self.simulator_pid <= 0:
            return False
        return not is_zombie(self.simulator_pid)

    def closeSimulator(self):
        """
        Close the simulator process using the existing kill_process function
        """
        if self.simulator_pid > 0:
            print("Closing the simulator with pid ", self.simulator_pid)
            kill_process(self.simulator_pid)

    def resetBallAndRobotPositions(self, seed=None):
        """
        The utility function to generate the ball and robot positions in reset() function call

        Sample usage of generatePoses()
        Here we initialize all robots in their penalty area, but only the goalie should be in goal area
        """

        purposedInitialPose: Dict[str, Point] = {}

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        initialBallPose: Point = generatePoses(
            SoccerFieldAreas.centerCircle, 1, rng=rng
        )[0]

        area = SoccerFieldAreas.ownPenaltyArea.difference(SoccerFieldAreas.ownGoalArea)
        candidatePos: List[Point] = generatePoses(area, len(self.team1), rng=rng)

        for robot in self.team1:
            if robot == 1:
                purposedInitialPose[str(robot)] = generatePoses(
                    SoccerFieldAreas.ownGoalArea, 1, rng=rng
                )[0]
            else:
                purposedInitialPose[str(robot)] = candidatePos.pop()

        area = SoccerFieldAreas.opponentPenaltyArea.difference(
            SoccerFieldAreas.opponentGoalArea
        )
        candidatePos = generatePoses(area, len(self.team2), rng=rng)

        for robot in self.team2:
            if robot == 21:
                purposedInitialPose[str(robot)] = generatePoses(
                    SoccerFieldAreas.opponentGoalArea, 1, rng=rng
                )[0]
            else:
                purposedInitialPose[str(robot)] = candidatePos.pop()

        # assign rotation
        for robot, robotPos in purposedInitialPose.items():
            rot = math.atan2(
                initialBallPose.y - robotPos.y, initialBallPose.x - robotPos.x
            )
            purposedInitialPose[robot] = [
                robotPos.x,
                robotPos.y,
                350.0,
                0.0,
                0.0,
                rot,
            ]
        purposedInitialPose["Ball"] = [initialBallPose.x, initialBallPose.y, 50.0]

        return purposedInitialPose

    def calcReward(self, extendedInfo: Dict[str, Any]) -> float:
        """
        Calculate the reward for the agent from extended info, which include some groundTruth information of the game status
        """
        return 1

    def configShm(self):
        """
        Configure the shared memory for communication to the simulator

        But we don't connect it here
        """

        self.globalConfigShm = ShmemHeap(self.pythonEnvPrefix + "GlobalConfig")
        self.globalConfig = ShmemAccessor(self.globalConfigShm)

        self.initialPosesShm = ShmemHeap(self.pythonEnvPrefix + "InitialPoses")
        self.initialPoses = ShmemAccessor(self.initialPosesShm)

        # Global shm
        self.globalShmManager = SharedMemoryManager(
            self.pythonEnvPrefix[:-1] if not DEBUG else "DEBUG",
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

        # Robot individual shm
        self.robotShmManagers: Dict[int, SharedMemoryManager] = {
            robot: SharedMemoryManager(
                self.pythonEnvPrefix + "robot" + str(robot),
                [
                    ("ActionRequestFlagShm", (1,), SHM_VERBOSE),
                    ("ObsArrayShm", (OBS_SIZE,), SHM_VERBOSE),
                    ("ActionArrayShm", (ACT_SIZE,), SHM_VERBOSE),
                ],
            )
            for robot in self.agentRobots
        }

        self.robotExtendedInfoShmems: Dict[Any, ShmemHeap] = {
            robot: ShmemHeap(
                self.pythonEnvPrefix + "robot" + str(robot) + "_" + "ExtendedInfo"
            )
            for robot in self.agentRobots
        }
        self.robotExtendedInfos: Dict[Any, ShmemAccessor] = {
            robot: ShmemAccessor(self.robotExtendedInfoShmems[robot])
            for robot in self.agentRobots
        }

    def openShm(self):
        """Unlink(delete) the shared memory blocks"""

        self.globalConfigShm.create()
        self.initialPosesShm.create()

        self.globalConfig.set(self.environmentVariables[self.possible_agents[0]])

        self.globalShmManager.createTunnels()

        self.actionUpdatedFlagArrayShm.sendArray([0] * len(self.possible_agents))

        for robot in self.agentRobots:
            self.robotShmManagers[robot].createTunnels()
            self.robotExtendedInfoShmems[robot].create()
            self.robotExtendedInfoShmems[robot].postCounterSem()

    def closeShm(self):
        """Close the shared memory blocks"""

        self.globalConfigShm.unlink()
        self.initialPosesShm.unlink()

        self.globalShmManager.unlink()

        for robot in self.agentRobots:
            self.robotShmManagers[robot].unlink()
            self.robotExtendedInfoShmems[robot].unlink()
