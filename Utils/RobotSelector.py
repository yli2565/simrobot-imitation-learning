from pettingzoo.utils import agent_selector, wrappers
from InterThreadCommunication import SharedMemoryHelper, SharedMemoryManager
from typing import Any, List, Dict, Set


class RobotSelector(agent_selector):
    """Outputs an agent in the given order whenever agent_select is called.

    Can reinitialize to a new order.

    Example:
        >>> from pettingzoo.utils import agent_selector
        >>> agent_selector = agent_selector(agent_order=["player1", "player2"])
        >>> agent_selector.reset()
        'player1'
        >>> agent_selector.next()
        'player2'
        >>> agent_selector.is_last()
        True
        >>> agent_selector.reinit()
        >>> agent_selector.next()
        'player2'
        >>> agent_selector.is_last()
        False
    """

    def __init__(
        self,
        possibleAgents: List[Any],
        robotObsShms: Dict[Any, SharedMemoryHelper],
        interruptCallback=lambda: False,
    ):
        self.possibleAgents: List[Any] = possibleAgents
        self.robotActionRequestFlagShms: Dict[SharedMemoryHelper] = robotObsShms
        self.interruptCallback = interruptCallback
        self.reinit()

    def reinit(self) -> None:
        """Recover the agent pool to indicate a new round"""
        self.agentPool: List[Any] = self.possibleAgents.copy()

    def reset(self) -> Any:
        """Reset to the original order."""
        self.reinit()
        return self.next()

    def next(self) -> Any:
        """Get the next agent."""
        if len(self.agentPool) == 0:
            self.reinit()

        # Just in case some agent get removed
        self.agentPool = list(
            set(self.agentPool).intersection(set(self.possibleAgents))
        )
        poolSize = len(self.agentPool)
        selectRobotIdx = 0
        if poolSize == 0:
            raise ValueError("Empty pool")
        while selectRobotIdx < len(self.agentPool):
            if poolSize != len(self.agentPool):
                return self.next()
            robot = self.agentPool[selectRobotIdx]
            if (
                self.robotActionRequestFlagShms[robot].probeSem() > 0
            ):  # This means that the robot need an action
                break
            if self.interruptCallback():
                break  # Just randomly pick one robot
            selectRobotIdx = (selectRobotIdx + 1) % poolSize
        self.agentPool.remove(robot)
        self.selected_agent = robot
        return self.selected_agent

    def is_last(self) -> bool:
        """Check if the current agent is the last agent in the cycle."""
        return len(self.agentPool) == 0

    def is_first(self) -> bool:
        """Check if the current agent is the first agent in the cycle."""
        return len(self.agentPool) == len(self.possibleAgents) - 1

    def __eq__(self, other: "RobotSelector") -> bool:
        if not isinstance(other, RobotSelector):
            return NotImplemented

        return (
            self.agentPool == other.agentPool
            and self.selected_agent == other.selected_agent
        )
