from enum import Enum, auto


# Define the Enums
class State(Enum):
    beforeHalf = 0
    afterHalf = auto()
    timeout = auto()
    playing = auto()
    setupOwnKickOff = auto()
    setupOpponentKickOff = auto()
    waitForOwnKickOff = auto()
    waitForOpponentKickOff = auto()
    ownKickOff = auto()
    opponentKickOff = auto()
    setupOwnPenaltyKick = auto()
    setupOpponentPenaltyKick = auto()
    waitForOwnPenaltyKick = auto()
    waitForOpponentPenaltyKick = auto()
    ownPenaltyKick = auto()
    opponentPenaltyKick = auto()
    ownPushingFreeKick = auto()
    opponentPushingFreeKick = auto()
    ownKickIn = auto()
    opponentKickIn = auto()
    ownGoalKick = auto()
    opponentGoalKick = auto()
    ownCornerKick = auto()
    opponentCornerKick = auto()
    beforePenaltyShootout = auto()
    waitForOwnPenaltyShot = auto()
    waitForOpponentPenaltyShot = auto()
    ownPenaltyShot = auto()
    opponentPenaltyShot = auto()
    afterOwnPenaltyShot = auto()
    afterOpponentPenaltyShot = auto()


class Phase(Enum):
    firstHalf = 0
    secondHalf = auto()
    penaltyShootout = auto()


class PlayerState(Enum):
    unstiff = 0
    calibration = auto()
    penalizedManual = auto()
    penalizedIllegalBallContact = auto()
    penalizedPlayerPushing = auto()
    penalizedIllegalMotionInSet = auto()
    penalizedInactivePlayer = auto()
    penalizedIllegalPosition = auto()
    penalizedLeavingTheField = auto()
    penalizedRequestForPickup = auto()
    penalizedLocalGameStuck = auto()
    penalizedIllegalPositionInSet = auto()
    penalizedPlayerStance = auto()
    substitute = auto()
    active = auto()


class GameControllerState(Enum):
    STATE_INITIAL = 0
    STATE_READY = auto()
    STATE_SET = auto()
    STATE_PLAYING = auto()
    STATE_FINISHED = auto()


class CompetitionPhase(Enum):
    roundRobin = 0
    playOff = auto()


class GameState:
    @staticmethod
    def isInitial(state: State) -> bool:
        return state in {State.beforeHalf, State.timeout, State.beforePenaltyShootout}

    @staticmethod
    def isReady(state: State) -> bool:
        return state in {
            State.setupOwnKickOff,
            State.setupOpponentKickOff,
            State.setupOwnPenaltyKick,
            State.setupOpponentPenaltyKick,
        }

    @staticmethod
    def isSet(state: State) -> bool:
        return state in {
            State.waitForOwnKickOff,
            State.waitForOpponentKickOff,
            State.waitForOwnPenaltyKick,
            State.waitForOpponentPenaltyKick,
            State.waitForOwnPenaltyShot,
            State.waitForOpponentPenaltyShot,
        }

    @staticmethod
    def isPlaying(state: State) -> bool:
        return state in {
            State.playing,
            State.ownKickOff,
            State.opponentKickOff,
            State.ownPenaltyKick,
            State.opponentPenaltyKick,
            State.ownPushingFreeKick,
            State.opponentPushingFreeKick,
            State.ownKickIn,
            State.opponentKickIn,
            State.ownGoalKick,
            State.opponentGoalKick,
            State.ownCornerKick,
            State.opponentCornerKick,
            State.ownPenaltyShot,
            State.opponentPenaltyShot,
        }

    @staticmethod
    def isFinished(state: State) -> bool:
        return state in {
            State.afterHalf,
            State.afterOwnPenaltyShot,
            State.afterOpponentPenaltyShot,
        }

    @staticmethod
    def isStopped(state: State) -> bool:
        return state in {
            State.beforeHalf,
            State.afterHalf,
            State.timeout,
            State.beforePenaltyShootout,
            State.afterOwnPenaltyShot,
            State.afterOpponentPenaltyShot,
        }

    @staticmethod
    def isKickOff(state: State) -> bool:
        return state in {
            State.setupOwnKickOff,
            State.setupOpponentKickOff,
            State.waitForOwnKickOff,
            State.waitForOpponentKickOff,
            State.ownKickOff,
            State.opponentKickOff,
        }

    @staticmethod
    def isPenaltyKick(state: State) -> bool:
        return state in {
            State.setupOwnPenaltyKick,
            State.setupOpponentPenaltyKick,
            State.waitForOwnPenaltyKick,
            State.waitForOpponentPenaltyKick,
            State.ownPenaltyKick,
            State.opponentPenaltyKick,
        }

    @staticmethod
    def isFreeKick(state: State) -> bool:
        return state in {
            State.ownPushingFreeKick,
            State.opponentPushingFreeKick,
            State.ownKickIn,
            State.opponentKickIn,
            State.ownGoalKick,
            State.opponentGoalKick,
            State.ownCornerKick,
            State.opponentCornerKick,
        }

    @staticmethod
    def isPushingFreeKick(state: State) -> bool:
        return state in {State.ownPushingFreeKick, State.opponentPushingFreeKick}

    @staticmethod
    def isKickIn(state: State) -> bool:
        return state in {State.ownKickIn, State.opponentKickIn}

    @staticmethod
    def isGoalKick(state: State) -> bool:
        return state in {State.ownGoalKick, State.opponentGoalKick}

    @staticmethod
    def isCornerKick(state: State) -> bool:
        return state in {State.ownCornerKick, State.opponentCornerKick}

    @staticmethod
    def isPenaltyShootout(state: State) -> bool:
        return state in {
            State.beforePenaltyShootout,
            State.waitForOwnPenaltyShot,
            State.waitForOpponentPenaltyShot,
            State.ownPenaltyShot,
            State.opponentPenaltyShot,
            State.afterOwnPenaltyShot,
            State.afterOpponentPenaltyShot,
        }

    @staticmethod
    def isForOwnTeam(state: State) -> bool:
        return state in {
            State.setupOwnKickOff,
            State.waitForOwnKickOff,
            State.ownKickOff,
            State.setupOwnPenaltyKick,
            State.waitForOwnPenaltyKick,
            State.ownPenaltyKick,
            State.ownPushingFreeKick,
            State.ownKickIn,
            State.ownGoalKick,
            State.ownCornerKick,
            State.waitForOwnPenaltyShot,
            State.ownPenaltyShot,
            State.afterOwnPenaltyShot,
        }

    @staticmethod
    def isForOpponentTeam(state: State) -> bool:
        return state in {
            State.setupOpponentKickOff,
            State.waitForOpponentKickOff,
            State.opponentKickOff,
            State.setupOpponentPenaltyKick,
            State.waitForOpponentPenaltyKick,
            State.opponentPenaltyKick,
            State.opponentPushingFreeKick,
            State.opponentKickIn,
            State.opponentGoalKick,
            State.opponentCornerKick,
            State.waitForOpponentPenaltyShot,
            State.opponentPenaltyShot,
            State.afterOpponentPenaltyShot,
        }

    @staticmethod
    def isPenalized(player_state: PlayerState) -> bool:
        return player_state in {
            PlayerState.penalizedManual,
            PlayerState.penalizedIllegalBallContact,
            PlayerState.penalizedPlayerPushing,
            PlayerState.penalizedIllegalMotionInSet,
            PlayerState.penalizedInactivePlayer,
            PlayerState.penalizedIllegalPosition,
            PlayerState.penalizedLeavingTheField,
            PlayerState.penalizedRequestForPickup,
            PlayerState.penalizedLocalGameStuck,
            PlayerState.penalizedIllegalPositionInSet,
            PlayerState.penalizedPlayerStance,
            PlayerState.substitute,
        }
