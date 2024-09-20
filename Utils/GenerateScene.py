TEAM1_NUMBER = "5"
TEAM1_COLOR = "black"
TEAM1_GOALIE_COLOR = "purple"
TEAM2_NUMBER = "70"
TEAM2_COLOR = "red"
TEAM2_GOALIE_COLOR = "blue"
LOG_DATA_ONLY_CON_SAMPLE="""dr annotation
dr timing
for Upper dr representation:JPEGImage off
for Upper dr representation:CameraImage off
for Upper dr representation:BallPercept
for Upper dr representation:BallSpots
for Upper dr representation:BodyContour
for Upper dr representation:CameraInfo
for Upper dr representation:CameraMatrix
for Upper dr representation:CirclePercept
for Upper dr representation:FieldBoundary
for Upper dr representation:FrameInfo
for Upper dr representation:ImageCoordinateSystem
for Upper dr representation:LinesPercept
for Upper dr representation:ObstaclesFieldPercept
for Upper dr representation:ObstaclesImagePercept
for Upper dr representation:OdometryData
for Upper dr representation:PenaltyMarkPercept
for Lower dr representation:JPEGImage off
for Lower dr representation:CameraImage off
for Lower dr representation:BallPercept
for Lower dr representation:BallSpots
for Lower dr representation:BodyContour
for Lower dr representation:CameraInfo
for Lower dr representation:CameraMatrix
for Lower dr representation:CirclePercept
for Lower dr representation:FieldBoundary
for Lower dr representation:FrameInfo
for Lower dr representation:ImageCoordinateSystem
for Lower dr representation:LinesPercept
for Lower dr representation:ObstaclesFieldPercept
for Lower dr representation:ObstaclesImagePercept
for Lower dr representation:OdometryData
for Lower dr representation:PenaltyMarkPercept
for Cognition dr representation:ActivationGraph
for Cognition dr representation:AlternativeRobotPoseHypothesis
for Cognition dr representation:ArmMotionRequest
for Cognition dr representation:BallModel
for Cognition dr representation:BehaviorStatus
for Cognition dr representation:CameraCalibration
for Cognition dr representation:FieldBall
for Cognition dr representation:FrameInfo
for Cognition dr representation:GameControllerData
for Cognition dr representation:GameState
for Cognition dr representation:GlobalOpponentsModel
for Cognition dr representation:GlobalTeammatesModel
for Cognition dr representation:IMUCalibration
for Cognition dr representation:HeadMotionRequest
for Cognition dr representation:MotionRequest
for Cognition dr representation:ObstacleModel
for Cognition dr representation:OdometryData
for Cognition dr representation:ReceivedTeamMessages
for Cognition dr representation:RobotHealth
for Cognition dr representation:RobotPose
for Cognition dr representation:SelfLocalizationHypotheses
for Cognition dr representation:SideInformation
for Cognition dr representation:SkillRequest
for Cognition dr representation:StrategyStatus
for Cognition dr representation:TeammatesBallModel
for Cognition dr representation:TeamData
for Cognition dr representation:WalkingEngineOutput
for Motion dr representation:FallDownState
for Motion dr representation:FootOffset
for Motion dr representation:FootSupport
for Motion dr representation:FrameInfo
for Motion dr representation:FsrData
for Motion dr representation:FsrSensorData
for Motion dr representation:GroundContactState
for Motion dr representation:GyroState
for Motion dr representation:InertialSensorData
for Motion dr representation:InertialData
for Motion dr representation:JointAnglePred
for Motion dr representation:JointCalibration
for Motion dr representation:JointPlay
for Motion dr representation:JointRequest
for Motion dr representation:JointSensorData
for Motion dr representation:KeyStates
for Motion dr representation:MotionInfo
for Motion dr representation:OdometryData
for Motion dr representation:OdometryDataPreview
for Motion dr representation:SystemSensorData
for Motion dr representation:RobotStableState
for Motion dr representation:WalkLearner
for Motion dr representation:WalkStepData
for Audio dr representation:AudioData
for Audio dr representation:FrameInfo
for Audio dr representation:Whistle
for Referee dr representation:RefereePercept
for Referee dr representation:Keypoints"""

LOG_ALL_CON_SAMPLE = """dr annotation
dr timing
for Upper dr representation:JPEGImage
for Upper dr representation:CameraImage
for Upper dr representation:BallPercept
for Upper dr representation:BallSpots
for Upper dr representation:BodyContour
for Upper dr representation:CameraInfo
for Upper dr representation:CameraMatrix
for Upper dr representation:CirclePercept
for Upper dr representation:FieldBoundary
for Upper dr representation:FrameInfo
for Upper dr representation:ImageCoordinateSystem
for Upper dr representation:LinesPercept
for Upper dr representation:ObstaclesFieldPercept
for Upper dr representation:ObstaclesImagePercept
for Upper dr representation:OdometryData
for Upper dr representation:PenaltyMarkPercept
for Lower dr representation:JPEGImage
for Lower dr representation:CameraImage
for Lower dr representation:BallPercept
for Lower dr representation:BallSpots
for Lower dr representation:BodyContour
for Lower dr representation:CameraInfo
for Lower dr representation:CameraMatrix
for Lower dr representation:CirclePercept
for Lower dr representation:FieldBoundary
for Lower dr representation:FrameInfo
for Lower dr representation:ImageCoordinateSystem
for Lower dr representation:LinesPercept
for Lower dr representation:ObstaclesFieldPercept
for Lower dr representation:ObstaclesImagePercept
for Lower dr representation:OdometryData
for Lower dr representation:PenaltyMarkPercept
for Cognition dr representation:ActivationGraph
for Cognition dr representation:AlternativeRobotPoseHypothesis
for Cognition dr representation:ArmMotionRequest
for Cognition dr representation:BallModel
for Cognition dr representation:BehaviorStatus
for Cognition dr representation:CameraCalibration
for Cognition dr representation:FieldBall
for Cognition dr representation:FrameInfo
for Cognition dr representation:GameControllerData
for Cognition dr representation:GameState
for Cognition dr representation:GlobalOpponentsModel
for Cognition dr representation:GlobalTeammatesModel
for Cognition dr representation:IMUCalibration
for Cognition dr representation:HeadMotionRequest
for Cognition dr representation:MotionInfo
for Cognition dr representation:MotionRequest
for Cognition dr representation:ObstacleModel
for Cognition dr representation:OdometryData
for Cognition dr representation:ReceivedTeamMessages
for Cognition dr representation:RobotHealth
for Cognition dr representation:RobotPose
for Cognition dr representation:SelfLocalizationHypotheses
for Cognition dr representation:SideInformation
for Cognition dr representation:SkillRequest
for Cognition dr representation:StrategyStatus
for Cognition dr representation:TeammatesBallModel
for Cognition dr representation:TeamData
for Cognition dr representation:WalkStepData
for Cognition dr representation:WalkingEngineOutput
for Motion dr representation:FallDownState
for Motion dr representation:FootOffset
for Motion dr representation:FootSupport
for Motion dr representation:FrameInfo
for Motion dr representation:FsrData
for Motion dr representation:FsrSensorData
for Motion dr representation:GroundContactState
for Motion dr representation:GyroState
for Motion dr representation:InertialSensorData
for Motion dr representation:InertialData
for Motion dr representation:JointAnglePred
for Motion dr representation:JointCalibration
for Motion dr representation:JointPlay
for Motion dr representation:JointRequest
for Motion dr representation:JointSensorData
for Motion dr representation:KeyStates
for Motion dr representation:MotionInfo
for Motion dr representation:OdometryData
for Motion dr representation:OdometryDataPreview
for Motion dr representation:SystemSensorData
for Motion dr representation:RobotStableState
for Motion dr representation:WalkLearner
for Motion dr representation:WalkStepData
for Audio dr representation:AudioData
for Audio dr representation:FrameInfo
for Audio dr representation:Whistle
for Referee dr representation:RefereePercept
for Referee dr representation:Keypoints"""

SCENE_CON_SAMPLE = """
call Includes/Normal

# all views are defined in another script
call Includes/Views

# press any key to activate the joystick
jc press 1 call Includes/Joystick
jc press 2 call Includes/Joystick
jc press 3 call Includes/Joystick
jc press 4 call Includes/Joystick

dr module:SelfLocator:activateSampleResettingToGroundTruth
call Includes/{log_con_filename}

# disable most functionality of auto referee
ar placeBall off
ar placePlayers off
ar switchToFinished off 
ar penalizeLeavingTheField off
ar penalizeIllegalPosition off
ar penalizeIllegalPositionInSet off
ar freeKickComplete off
ar unpenalize off

# enable simulated time
dt off
st on

gc ready
"""

ROS2_SAMPLE = """
<Simulation>

  <Include href="Includes/NaoV6H25.rsi2"/>
  <Include href="Includes/Ball2016SPL.rsi2"/>
  <Include href="Includes/Field2020SPL.rsi2"/>

  <Scene name="RoboCup" controller="SimulatedNao" stepLength="0.012" color="rgb(65%, 65%, 70%)" ERP="0.8" CFM="0.001" contactSoftERP="0.2" contactSoftCFM="0.005">
    <PointLight z="9m" ambientColor="rgb(50%, 50%, 50%)"/>

    <Compound name="teams">
        <Compound name="B-Human">
        <Compound name="{team1_number}"/>
        <Compound name="{team1_color}"/>
        <Compound name="{team1_goalie_color}"/>
        </Compound>
        <Compound name="B-Team">
        <Compound name="{team2_number}"/>
        <Compound name="{team2_color}"/>
        <Compound name="{team2_goalie_color}"/>
        </Compound>
    </Compound>

    <Compound name="robots">
        {robots}
    </Compound>

    <Compound name="extras">
        {dummy_robots}
    </Compound>


    <Compound name="balls">
      <Body ref="ball">
        <Translation x="-3.2" y="0" z="1m"/>
      </Body>
    </Compound>

    <Compound ref="field"/>

  </Scene>
</Simulation>
"""
ROBOT_SAMPLE = """
    <Body ref="Nao" name="robot{robot_id}">
      <Translation x="{x}cm" y="{y}cm" z="320mm"/>
      <Rotation z="{rotation}degree"/>
      <Set name="NaoColor" value="{color}"/>
    </Body>
"""
DUMMY_ROBOT_SAMPLE = """
    <Body ref="NaoDummy" name="robot{robot_id}">
      <Translation x="{x}cm" y="{y}cm" z="320mm"/>
      <Rotation z="{rotation}degree"/>
      <Set name="NaoColor" value="{color}"/>
    </Body>
"""


def generateSceneCon(LogConFileName: str):
    if not LogConFileName.endswith(".con"):
        LogConFileName += ".con"
    return SCENE_CON_SAMPLE.format(log_con_filename=LogConFileName)

def generateRos2(
    robots,
    dummyRobots,
    team1Number=TEAM1_NUMBER,
    team1Color=TEAM1_COLOR,
    team1GoalieColor=TEAM1_GOALIE_COLOR,
    team2Number=TEAM2_NUMBER,
    team2Color=TEAM2_COLOR,
    team2GoalieColor=TEAM2_GOALIE_COLOR,
):
    robotsXML = ""
    for idx,robot in enumerate(robots):
        idx+=1
        if robot == 1:
            robotsXML += ROBOT_SAMPLE.format(
                robot_id=robot, x=-idx*25, y=-idx*25, rotation=0, color=team1GoalieColor
            )
        elif robot in range(2, 20):
            robotsXML += ROBOT_SAMPLE.format(
                robot_id=robot, x=-idx*25, y=-idx*25, rotation=0, color=team1Color
            )
        elif robot == 21:
            robotsXML += ROBOT_SAMPLE.format(
                robot_id=robot, x=-idx*25, y=-idx*25, rotation=0, color=team2GoalieColor
            )
        elif robot in range(20, 40):
            robotsXML += ROBOT_SAMPLE.format(
                robot_id=robot, x=-idx*25, y=-idx*25, rotation=0, color=team2Color
            )
        else:
            raise Exception("Invalid robot number")
    dummyRobotsXML = ""
    for idx,robot in enumerate(dummyRobots):
        idx+=1
        if robot == 1:
            dummyRobotsXML += DUMMY_ROBOT_SAMPLE.format(
                robot_id=robot, x=idx*25, y=idx*25, rotation=0, color=team1GoalieColor
            )
        elif robot in range(2, 20):
            dummyRobotsXML += DUMMY_ROBOT_SAMPLE.format(
                robot_id=robot, x=idx*25, y=idx*25, rotation=0, color=team1Color
            )
        elif robot == 21:
            dummyRobotsXML += DUMMY_ROBOT_SAMPLE.format(
                robot_id=robot, x=idx*25, y=idx*25, rotation=0, color=team2GoalieColor
            )
        elif robot in range(20, 40):
            dummyRobotsXML += DUMMY_ROBOT_SAMPLE.format(
                robot_id=robot, x=idx*25, y=idx*25, rotation=0, color=team2Color
            )
        else:
            raise Exception("Invalid dummy robot number")
    simulationXML = ROS2_SAMPLE.format(
        robots=robotsXML,
        dummy_robots=dummyRobotsXML,
        team1_number=team1Number,
        team1_color=team1Color,
        team1_goalie_color=team1GoalieColor,
        team2_number=team2Number,
        team2_color=team2Color,
        team2_goalie_color=team2GoalieColor,
    )
    return simulationXML

def generateLogCon():
    return LOG_DATA_ONLY_CON_SAMPLE

if __name__ == "__main__":
    print(generateRos2([5], [21]))
