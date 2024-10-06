from __future__ import annotations

import functools
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union, cast

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from overrides import overrides
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ActionType, AgentID, ObsType

from InterThreadCommunication import SharedMemoryHelper, SharedMemoryManager
from Utils import (
    ObservationJosh,
    cfg2dict,
    kill_process,
    is_zombie,
    generateRos2,
    generateSceneCon,
)

sys.path.append(sys.path[0] + "/..")

# these paths must match the ones used by the c++ code
# BADGER_RL_SYSTEM_DIR = Path("/root/autodl-tmp/BadgerRLSystem/")
BADGER_RL_SYSTEM_DIR = Path("/home/yuhao2024/Documents/BadgerRLSystem-josh/")
# BADGER_RL_SYSTEM_DIR = Path("/home/chkxwlyh/Documents/Study/RL100/BadgerRLSystem/")

DEBUG_PRINTS = False
DEBUG = True

RAW_OBS_SIZE = 100
OBS_SIZE = 45
RAW_ACT_SIZE = 20
ACT_SIZE = 2
INFO_SIZE = 55
WORLD_GROUND_TRUTH_SIZE = 4
ROBOT_GROUND_TRUTH_SIZE = 3
EPISODE_TIME_LIMIT = 60 * 1000  # (ms)
SHM_VERBOSE = 2
CALC_OBS_IN_PYTHON = False
CALC_ACT_IN_PYTHON = False
CALC_REWARD_IN_PYTHON = True


def env(render_mode=None, curriculum_method="close"):
    env = parallel_env(curriculum_method=curriculum_method)
    return env


class parallel_env(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode="rgb_array", curriculum_method="close"):
        self.rendering_init = False
        self.render_mode = render_mode
        self.curriculum_method = curriculum_method

        self.randSeed = 0

        # counters
        self.episode_count = 0
        self.episode_step_count = 0
        self.global_step_count = 0

        # Scene
        self.sceneName = "PythonEnvGenerated"

        # agents
        # Team 1's robot is in [0, 19], Team 2's robot is in [20, 39]
        # The agent number of a robot is its robot number - 1. For example, robot 1 is agent 0

        # Each robot is controlled by a RL agent
        self.robots = [4, 5, 24]
        # Each dummy robot is just an obstacle. Their position can be set during reset
        self.dummyRobots = [1, 21]
        self.BhumanRobots = []  # TODO: Implement robots following Bhuman code
        self.HijectedRobots = []  # TODO: Implement robots following hard coded logic

        self.team0 = list(filter(lambda x: 0 < x < 20, self.robots))
        self.team1 = list(filter(lambda x: 20 < x < 40, self.robots))
        self.teams = {0: self.team0, 1: self.team1}

        overlaps = set(self.robots) & set(self.dummyRobots)
        if bool(overlaps):
            raise ValueError(
                f"Robot(s): {overlaps} are both dummy and active, impossible"
            )

        self.possible_agents = [r - 1 for r in self.robots]
        self.possible_agents = cast(List[AgentID], self.possible_agents)
        self.agents: List[AgentID] = self.possible_agents[:]

        # RL data calculator
        self.RLDataCalc: Dict[int, ObservationJosh] = {
            agent: ObservationJosh("WalkToBall") for agent in self.agents
        }

        # 6D vector of (x, y, angle, kick_strength) velocity changes
        action_space: gym.spaces.Space[ActionType] = gym.spaces.Box(
            np.array([-1] * ACT_SIZE),
            np.array([1] * ACT_SIZE),
        )
        self.action_spaces = {agent: action_space for agent in self.agents}

        observation_space: gym.spaces.Space[ObsType] = gym.spaces.Box(
            low=-1, high=1, shape=(OBS_SIZE,)
        )
        self.observation_spaces = {agent: observation_space for agent in self.agents}

        self.fetchedInitialObs = None

        self._openShm()

    def __del__(self):
        self.close()

    # Core function (Should be implemented either in C++ or Python)
    def calcObs(self, robot, rawObs, robotInfo) -> ObsType:
        """
        If you decide to implement this function, it means you want to calculate obs
        in Python, the rawObs is created in updateRawObservation() function in C++

        Default:
        rawObs {ball x, ball y, robot x, robot y, robot rotation, m = num of teammates, n = num of opponents, m*2 teammates (x,y), n*2 opponents (x,y)}

        If you want to calculate obs in C++, set CALC_OBS_IN_PYTHON = False
        and implement updateUserDefinedObservation() in NeuralControl.cpp

        EITHER THIS FUNCTION OR updateUserDefinedObservation() SHOULD BE IMPLEMENTED
        """
        agent = robot - 1
        result = self.RLDataCalc[agent].getObservation(rawObs[2:5], rawObs[0:2])
        return result
        # raise NotImplementedError("calcUserDefinedObs not implemented")

    def calcAction(self, robot, inferredAction) -> ActionType:
        """
        Convert inferred action from neural network into BHuman API action
        """
        raise NotImplementedError("calcUserDefinedAction not implemented")
        pass

    def processInfo(self, rawInfo) -> Dict:
        """
        Due to limitation of my shared memory helper, we cannot send dict directly from
        C++. We need to parse the float array into dict here

        This info dict is just an example
        """
        return (
            {"frameElapse": rawInfo[0], "obsInCXX": rawInfo[1 : 1 + OBS_SIZE]}
            if rawInfo is not None
            else None
        )

    def calcReward(self, robot, syncGroundTruth, info) -> float:
        """
        Generally, it is highly recommended to calculate reward based on ground truth
        information only.

        If you want to calculate reward in C++, set CALC_REWARD_IN_PYTHON = False
        and implement updateReward() in NeuralControl.cpp

        If you need more ground truth information, you can get them from game
        controller (GameController.cpp) and send them in sendGroundTruthInformation()

        EITHER THIS FUNCTION OR updateReward() SHOULD BE IMPLEMENTED

        This reward is just an example
        """
        agent = robot - 1
        agentIdx = self.agents.index(agent)

        robotGroundTruthInfo = syncGroundTruth[WORLD_GROUND_TRUTH_SIZE:]
        worldGroundTruthInfo = syncGroundTruth[:WORLD_GROUND_TRUTH_SIZE]

        robotGroundTruthIndex = agentIdx * ROBOT_GROUND_TRUTH_SIZE
        robotGroundTruth = robotGroundTruthInfo[
            robotGroundTruthIndex : robotGroundTruthIndex + ROBOT_GROUND_TRUTH_SIZE
        ]
        ballOutType = worldGroundTruthInfo[2]
        episodeTime = worldGroundTruthInfo[3]

        team1OutOfBounds = ballOutType == 4
        team2OutOfBounds = ballOutType == 3
        team1Goal = ballOutType == 2
        team2Goal = ballOutType == 1
        realBallLoc = worldGroundTruthInfo[0:2]
        realAgentLoc = robotGroundTruth[0:3]

        Goal = team2Goal
        OutOfBounds = team2OutOfBounds
        # Convert to robot's own coordinate
        if robot in self.teams[0]:
            realAgentLoc[0] = realAgentLoc[0] * -1
            realAgentLoc[1] = realAgentLoc[1] * -1
            realBallLoc[0] = realBallLoc[0] * -1
            realBallLoc[1] = realBallLoc[1] * -1

            Goal = team1Goal
            OutOfBounds = team1OutOfBounds

        realBallLoc = realBallLoc
        reward = self.RLDataCalc[agent].getReward(
            realAgentLoc, realBallLoc, info["frameElapse"]
        )

        return reward

    def calcResetPose(self):
        """
        Reset Pose array layout
        [ball_x, ball_y,
        robot_1_x, robot_1_y, robot_1_z, robot_1_x_rot, robot_1_y_rot, robot_1_z_rot,
        robot_2_x, robot_2_y, robot_2_z, robot_2_x_rot, robot_2_y_rot, robot_2_z_rot, ...]

        Dummy Reset Pose array layout
        [robot_1_x, robot_1_y, robot_1_z, robot_1_x_rot, robot_1_y_rot, robot_1_z_rot,
        robot_2_x, robot_2_y, robot_2_z, robot_2_x_rot, robot_2_y_rot, robot_2_z_rot, ...]

        TIP: you can access field information in "Config/Locations/Default/fieldDimensions.cfg" with
        cfg2dict(BADGER_RL_SYSTEM_DIR / "Config/Locations/Default/fieldDimensions.cfg")
        """
        dummyResetPosArray = [
            -100
        ] * self.dummyResetPosShm.arraySize  # -100 means keep the same
        resetPosArray = [-100] * self.resetPosShm.arraySize  # -100 means keep the same
        ballPos = randomBallPos()
        resetPosArray[0] = ballPos[0]
        resetPosArray[1] = ballPos[1]
        for robotIdx in range(len(self.robots)):
            robotPos = randomRobotPos(ballPos)
            for i in range(6):
                resetPosArray[2 + robotIdx * 6 + i] = robotPos[i]

        for dummyIdx in range(len(self.dummyRobots)):
            dummyRobotPos = randomRobotPos(ballPos)
            for i in range(6):
                dummyResetPosArray[dummyIdx * 6 + i] = dummyRobotPos[i]

        return resetPosArray, dummyResetPosArray

    # The following are functions to communicate with SimRobot
    # Before you make modification to these functions, please notice:
    # The Bhuman code runs each simulated robot in a separate thread
    # but update them in a synchronized way. (Robot 1 -> Robot 2 -> ...)
    # So you can wait on python side but don't use any blocking wait
    # in Neural Control.

    # For example, Some state of GameController would wait until all
    # robots are ready, if one robot is waiting for something else
    # from python side, it will cause deadlock.

    @overrides
    def reset(
        self,
        seed: Union[int, None] = None,
        options: Union[dict, None] = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        print_debug("Reset SimRobot")

        initialObs = {}
        initialInfo = {}

        if not hasattr(self, "sim_pid") or self.sim_pid is None:
            self.startSimRobot()

        self.exitFlagShm[0] = 1
        for robot in self.robots:
            self.robotShmManagers[robot]["observation_shm"].clearSem()

        resetPosArray, dummyResetPosArray = self.calcResetPose()
        self.resetPosShm.sendArray(resetPosArray)
        if len(self.dummyRobots) > 0:
            self.dummyResetPosShm.sendArray(dummyResetPosArray)

        if self.fetchedInitialObs is not None:
            # Duplicated reset, we already get the first obs
            initialObs = self.fetchedInitialObs
        else:
            # First, ensure the game controller finish resetting robots & ball positions
            while self.gcModeFlagShm[0] != 2:
                pass
            # Then, ensure all agents send the first obs
            obsFetched = np.array([False] * len(self.robots))
            agentIdx = 0
            while not obsFetched.all():
                agent = self.agents[agentIdx]
                robot = agent + 1
                if not obsFetched[agentIdx]:
                    processedObs, processedInfo = self.getObsAndInfo(robot)
                    if processedObs is not None:
                        initialObs[agent] = processedObs
                        initialInfo[agent] = processedInfo
                        obsFetched[agentIdx] = True
                    else:
                        if self.sim_pid != -1 and is_zombie(self.sim_pid):
                            kill_process(self.sim_pid)
                            raise Exception("Simulator shutdown unexpectedly")

                agentIdx = (agentIdx + 1) % len(self.robots)

            self.episode_count += 1

        self._clearShms()

        self.episode_step_count = 0

        self.fetchedInitialObs = initialObs

        print_debug(f"Starting episode {self.episode_count}")

        return initialObs, initialInfo

    @overrides
    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],  # observation dict
        dict[AgentID, float],  # reward dict
        dict[AgentID, bool],  # terminated dict
        dict[AgentID, bool],  # truncated dict
        dict[AgentID, dict],  # info dict
    ]:
        self.exitFlagShm[0] = 0  # Remove exit flag
        """
        step(action) takes in an action for each agent and should return the
        - observations  
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        robotSyncGroundTruth = {}
        self.fetchedInitialObs = None  # Once a step is token, remove the initial obs to indicate start of an episode

        self.episode_step_count += 1
        self.global_step_count += 1
        print_debug(
            f"Episode {self.episode_count} step {self.episode_step_count} (global step {self.global_step_count})"
        )

        for agent in self.agents:
            robot = agent + 1
            print_debug(f"Write action for agent {robot}: {actions[agent]}...")
            inferredAction = actions[agent]

            # It's confusing here, raw actually means it can directly interact with BHuman API, maybe there's a better name
            if CALC_ACT_IN_PYTHON:
                BhumanAction = self.calcAction(robot, inferredAction)
                self.robotShmManagers[robot]["raw_act_shm"].sendArray(BhumanAction)
            else:
                self.robotShmManagers[robot][
                    "raw_act_shm"
                ].sem.release()  # Please keep this line for synchronization
                self.robotShmManagers[robot]["action_shm"].sendArray(actions[agent])

        # Count the time between two actions
        if not hasattr(self, "now"):
            self.now = time.time()
        else:
            delta_time = time.time() - self.now
            print(f"Delta time between two actions (ms): {delta_time*1000}")
        self.now = time.time()

        obsFetched = np.array([False] * len(self.robots))
        agentIdx = 0
        while not obsFetched.all():
            # Update termination and truncation flags
            terminatedFlag = self.terminatedFlagShm[0] == 1.0
            truncatedFlag = self.truncatedFlagShm[0] == 1.0

            # Even should terminate/truncate, fetch one more observation as the end obs
            agentIdx = (agentIdx + 1) % len(self.robots)
            agent = self.agents[agentIdx]
            robot = agent + 1
            if not obsFetched[agentIdx]:
                processedObs, processedInfo = self.getObsAndInfo(
                    robot
                )  # Fetch observation without blocking
                if processedObs is not None:
                    rew[agent], robotSyncGroundTruth[agent] = (
                        self.getRewardAndGroundTruth(robot, processedInfo)
                    )
                    obs[agent] = processedObs
                    info[agent] = processedInfo
                    obsFetched[agentIdx] = True
                else:
                    if self.sim_pid != -1 and is_zombie(self.sim_pid):
                        kill_process(self.sim_pid)
                        raise Exception("Simulator shutdown unexpectedly")

        # DONE: Now these are just debug info (instantaneous, not synced with robot obs)
        # worldGroundTruthInfo = self.worldGroundTruthShm.probeArray()
        # robotGroundTruthInfo = self.robotGroundTruthShm.probeArray()

        for agent in self.agents:
            truncated[agent] = truncatedFlag
            terminated[agent] = terminatedFlag
        self.now = time.time()
        return obs, rew, terminated, truncated, info

    def getObsAndInfo(self, robotNumber) -> Tuple[ObsType, Dict]:
        """
        This is just a wrapper to unify observation calculated in python and C++
        Don't write logic here
        """
        fetchedRawObs = None
        fetchedObs = None
        fetchedInfo = None
        processedInfo = None
        rawObsShm = self.robotShmManagers[robotNumber]["raw_obs_shm"]
        obsShm = self.robotShmManagers[robotNumber]["observation_shm"]
        infoShm = self.robotShmManagers[robotNumber]["info_shm"]
        # raw obs would always be the first to send, so if it's not available, others are also not available
        if rawObsShm.probeSem() > 0:
            fetchedRawObs = rawObsShm.fetchArray()
            fetchedInfo = infoShm.fetchArray()
            processedInfo = self.processInfo(fetchedInfo)

            if CALC_OBS_IN_PYTHON:
                processedObs = self.calcObs(robotNumber, fetchedRawObs, processedInfo)
            else:
                if obsShm.probeSem() > 0:
                    fetchedObs = obsShm.fetchArray()
                else:
                    raise ValueError(
                        f"Robot{robotNumber}'s obs is not synced with raw obs"
                    )
                processedObs = fetchedObs
            return processedObs, processedInfo
        else:
            return None, None  # The obs is not available yet

    def getRewardAndGroundTruth(self, robot, info) -> Tuple[float, Iterable]:
        syncGroundTruth = self.robotShmManagers[robot][
            "sync_ground_truth_shm"
        ].fetchArray()

        if CALC_REWARD_IN_PYTHON:
            reward = self.calcReward(robot, syncGroundTruth, info)
        else:
            reward = self.robotShmManagers[robot]["reward_shm"].fetchArray()[0]
        return reward, syncGroundTruth

    @overrides
    def render(self) -> Union[None, np.ndarray, str, list]:
        """Don't render"""
        pass

    def close(self) -> None:
        # Kill the SimRobot process
        super(parallel_env, self).close()
        if hasattr(self, "sim_pid"):
            print(f"Closing SimRobot Process: {self.sim_pid}")
            os.kill(self.sim_pid, signal.SIGTERM)

        # Launch Simulator

    def startSimRobot(self):
        with open(
            BADGER_RL_SYSTEM_DIR / f"Config/Scenes/{self.sceneName}.ros2", "w"
        ) as ros2File:
            for robot in self.robots:
            ros2File.write(generateRos2(self.robots, self.dummyRobots))
        with open(
            BADGER_RL_SYSTEM_DIR / f"Config/Scenes/{self.sceneName}.con", "w"
        ) as conFile:
            conFile.write(generateSceneCon())

        if DEBUG:
            self.sim_pid = -1
            return
        print_debug("Start CPP simulator")

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

            runCommand = [
                str(BADGER_RL_SYSTEM_DIR / "Build/Linux/SimRobot/Release/SimRobot"),
                "-g",
                str(BADGER_RL_SYSTEM_DIR / f"Config/Scenes/{self.sceneName}.ros2"),
            ]
            # This command will launch the simulator in the background (separate process)
            # process = subprocess.Popen(
            #     runCommand, cwd=BADGER_RL_SYSTEM_DIR, preexec_fn=os.setsid
            # )
            # You can enable printing from BadgerRLSystem by stdout=subprocess.DEVNULL -> stdout=subprocess.PIPE
            env = os.environ.copy()
            if not DEBUG:
                env["PYTHON_ENV_PID"] = str(os.getpid())
            # process = subprocess.Popen(
            #     runCommand,
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.PIPE,
            #     cwd=BADGER_RL_SYSTEM_DIR,
            #     env=env,
            # )
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

        self.sim_pid = process.pid
        return

    # Shared Memory
    def _openShm(self):
        if DEBUG:
            pythonEnvPrefix = ""
        else:
            pythonEnvPID = os.getpid()
            pythonEnvPrefix = str(pythonEnvPID) + "_"

        self.configShm = SharedMemoryHelper(
            pythonEnvPrefix + "config_shm", shape=(200,), verbose=0
        )
        self.configShm.createTunnel()
        config = [-100] * 200
        for robot in self.robots:
            config[robot] = 1
        for robot in self.dummyRobots:
            config[robot] = 0
        config[0] = 200  # The first element is config size
        config[45] = int(CALC_OBS_IN_PYTHON)
        config[46] = int(CALC_ACT_IN_PYTHON)
        config[47] = int(CALC_REWARD_IN_PYTHON)

        config[50] = RAW_OBS_SIZE
        config[51] = OBS_SIZE
        config[52] = RAW_ACT_SIZE
        config[53] = ACT_SIZE
        config[54] = INFO_SIZE
        config[55] = WORLD_GROUND_TRUTH_SIZE
        config[56] = ROBOT_GROUND_TRUTH_SIZE
        config[57] = EPISODE_TIME_LIMIT

        self.configShm.sendArray(config)

        self.robotShmManagers: Dict[int, SharedMemoryManager] = {
            robot: SharedMemoryManager(
                pythonEnvPrefix + "robot" + str(robot),
                [
                    ("raw_obs_shm", (RAW_OBS_SIZE,), SHM_VERBOSE),
                    ("observation_shm", (OBS_SIZE,), SHM_VERBOSE),
                    ("raw_act_shm", (RAW_ACT_SIZE,), SHM_VERBOSE),
                    ("action_shm", (ACT_SIZE,), SHM_VERBOSE),
                    (
                        "sync_ground_truth_shm",
                        (
                            WORLD_GROUND_TRUTH_SIZE
                            + ROBOT_GROUND_TRUTH_SIZE * len(self.robots),
                        ),
                        SHM_VERBOSE,
                    ),
                    ("reward_shm", (1,), SHM_VERBOSE),
                    ("info_shm", (INFO_SIZE,), SHM_VERBOSE),
                ],
            )
            for robot in self.robots
        }
        for rM in self.robotShmManagers.values():
            rM.createTunnels()

        self.globalShmManager = SharedMemoryManager(
            str(pythonEnvPID) if not DEBUG else "",
            [
                ("world_ground_truth_shm", (WORLD_GROUND_TRUTH_SIZE,), SHM_VERBOSE),
                (
                    "robot_ground_truth_shm",
                    (ROBOT_GROUND_TRUTH_SIZE * len(self.robots),),
                    SHM_VERBOSE,
                ),
                ("terminated_shm", (1,), SHM_VERBOSE),
                ("truncated_shm", (1,), SHM_VERBOSE),
                ("exit_shm", (1,), SHM_VERBOSE),
                ("reset_pos_shm", (2 + 6 * len(self.robots),), SHM_VERBOSE),
                ("dummy_reset_pos_shm", (6 * len(self.dummyRobots),), SHM_VERBOSE),
                ("robot_state_shm", (len(self.robots),), SHM_VERBOSE),
                ("gc_state_shm", (1,), SHM_VERBOSE),
                ("gc_mode_shm", (1,), SHM_VERBOSE),
            ],
            pythonEnvPrefix == "",
        )
        self.globalShmManager.createTunnels()
        # [0]: ball x, [1]: ball y; Currently only for debugging
        self.worldGroundTruthShm = self.globalShmManager["world_ground_truth_shm"]
        # robot_num * [x, y, rot]; Currently only for debugging
        self.robotGroundTruthShm = self.globalShmManager["robot_ground_truth_shm"]
        # 0 = False, 1 = True; Triggered by GC when ball out of bound (goal/out of bound)
        self.terminatedFlagShm = self.globalShmManager["terminated_shm"]
        # 0 = False, 1 = True; Triggered by GC when time out
        self.truncatedFlagShm = self.globalShmManager["truncated_shm"]
        # 0 = False, 1 = True; Triggered by Python env when termination/truncate is set
        self.exitFlagShm = self.globalShmManager["exit_shm"]
        # ball x, ball y, [x, y, z, x rot, y rot, z rot] ...
        # 2 + robot_num * 6
        self.resetPosShm = self.globalShmManager["reset_pos_shm"]
        # [x, y, z, x rot, y rot, z rot] ...
        # dummy_robot_num * 6
        self.dummyResetPosShm = self.globalShmManager["dummy_reset_pos_shm"]
        # The Robot FSM's state, for debugging
        self.robotStateShm = self.globalShmManager["robot_state_shm"]
        # The GC FSM's state, for debugging
        self.gcStateShm = self.globalShmManager["gc_state_shm"]
        # The state of GC: -1 = not initialized; 0 = resetting; 1 = running; 2 = waiting for robots
        self.gcModeFlagShm = self.globalShmManager["gc_mode_shm"]

        # Init shared memories
        for rM in self.robotShmManagers.values():
            # Init raw obs shm, since in some case we don't use the content but use it to synchronize python and C++
            # We don't want C++ side to receive something unpredictable
            rM["raw_obs_shm"].array[:] = 0

        self.worldGroundTruthShm.sendArray([-1] * self.worldGroundTruthShm.arraySize)
        self.robotGroundTruthShm.sendArray([-1] * self.robotGroundTruthShm.arraySize)
        self.terminatedFlagShm.sendArray([0])
        self.truncatedFlagShm.sendArray([0])  # 0 = False
        self.exitFlagShm.sendArray([0])  # 0 = False
        self.robotStateShm.sendArray([-10] * len(self.robots))
        self.gcStateShm.sendArray([-10])
        self.gcModeFlagShm.sendArray([-1])

        print_debug("Opened all shared memory")

    def _clearShms(self):
        self.worldGroundTruthShm.clearShm()
        self.robotGroundTruthShm.clearShm()
        for robot in self.robots:
            self.robotShmManagers[robot].clear(
                [
                    "raw_obs_shm",
                    "observation_shm",
                    "raw_act_shm",
                    "action_shm",
                    "info_shm",
                    "sync_ground_truth_shm",
                    "reward_shm",
                ]
            )

    def _closeShm(self):
        """
        Close all shared memory
        Actually all the Shm would automatically close in their __del__ ...
        """
        self.configShm.close()
        self.configShm.unlink()
        for robot in self.robots:
            self.robotShmManagers[robot].close()
            self.robotShmManagers[robot].unlink()
        self.globalShmManager.close()
        self.globalShmManager.unlink()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]


# Util functions
def print_debug(message: str) -> None:
    if DEBUG_PRINTS:
        print(message)


class BadgerRLSystemModifier:
    def __init__(self, badgerRLSystemDir):
        self.badgerRLSystemDir: Path = Path(badgerRLSystemDir)
        self.fieldDimensionsPath: Path = (
            self.badgerRLSystemDir / "Config/Locations/Default/fieldDimensions.cfg"
        )


FIELD_DIM = cfg2dict(
    BADGER_RL_SYSTEM_DIR / "Config/Locations/Default/fieldDimensions.cfg"
)


def randomBallPos():
    """[xBallPos, yBallPos]"""
    xBallPos = (
        random.random()
        * (FIELD_DIM["xPosOwnGroundLine"] - FIELD_DIM["xPosOwnPenaltyArea"])
        + FIELD_DIM["xPosOwnPenaltyArea"]
    )
    yBallPos = (
        random.random()
        * (FIELD_DIM["yPosLeftPenaltyArea"] - FIELD_DIM["yPosRightPenaltyArea"])
        + FIELD_DIM["yPosRightPenaltyArea"]
    )
    return xBallPos, yBallPos


def randomRobotPos(ballPos: Tuple[float, float], lookAtBall=True):
    """[x, y, z, x rot, y rot, z rot]"""
    xBallPos, yBallPos = ballPos
    while True:
        xRobotPos = random.random() * (FIELD_DIM["xPosOwnGroundLine"] + 2500) - 2500
        yRobotPos = (
            random.random()
            * (FIELD_DIM["yPosLeftSideline"] - FIELD_DIM["yPosRightSideline"])
            + FIELD_DIM["yPosRightSideline"]
        )

        if not (
            FIELD_DIM["xPosOwnGroundLine"] <= xRobotPos <= FIELD_DIM["xPosOwnGoalArea"]
            and FIELD_DIM["yPosRightGoalArea"]
            <= yRobotPos
            <= FIELD_DIM["yPosLeftGoalArea"]
        ):
            break

    rot = math.atan2(yBallPos - yRobotPos, xBallPos - xRobotPos)
    return xRobotPos, yRobotPos, 350.0, 0, 0, rot
