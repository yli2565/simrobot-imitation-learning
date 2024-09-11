TEAM1_COLOR = "black"
TEAM1_GOALIE_COLOR = "purple"
TEAM2_COLOR = "red"
TEAM2_GOALIE_COLOR = "blue"

CON_SAMPLE = """
call Includes/Normal

# all views are defined in another script
call Includes/Views

# press any key to activate the joystick
jc press 1 call Includes/Joystick
jc press 2 call Includes/Joystick
jc press 3 call Includes/Joystick
jc press 4 call Includes/Joystick

dr module:SelfLocator:activateSampleResettingToGroundTruth
call Includes/LogAsRealRobotDataOnly

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
gc set
"""
ROS2_SAMPLE="""
<Simulation>

  <Include href="Includes/NaoV6H25.rsi2"/>
  <Include href="Includes/Ball2016SPL.rsi2"/>
  <Include href="Includes/Field2020SPL.rsi2"/>

  <Scene name="RoboCup" controller="SimulatedNao" stepLength="0.012" color="rgb(65%, 65%, 70%)" ERP="0.8" CFM="0.001" contactSoftERP="0.2" contactSoftCFM="0.005">
    <PointLight z="9m" ambientColor="rgb(50%, 50%, 50%)"/>

    <Compound name="teams">
        <Compound name="B-Human">
        <Compound name="5"/>
        <Compound name="{team1_color}"/>
        <Compound name="{team1_goalie_color}"/>
        </Compound>
        <Compound name="B-Team">
        <Compound name="70"/>
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
      <Translation x="{x}" y="{y}" z="320mm"/>
      <Rotation z="{rotation}degree"/>
      <Set name="NaoColor" value="{color}"/>
    </Body>
"""
DUMMY_ROBOT_SAMPLE = """
    <Body ref="NaoDummy" name="robot{robot_id}">
      <Translation x="{x}" y="{y}" z="320mm"/>
      <Rotation z="{rotation}degree"/>
      <Set name="NaoColor" value="{color}"/>
    </Body>
"""
def generateCon():
    return CON_SAMPLE
def generateRos2(
    robots,
    dummyRobots,
    team1Color=TEAM1_COLOR,
    team1GoalieColor=TEAM1_GOALIE_COLOR,
    team2Color=TEAM2_COLOR,
    team2GoalieColor=TEAM2_GOALIE_COLOR,
):
    robotsXML = ""
    for robot in robots:
        if robot == 1:
            robotsXML += ROBOT_SAMPLE.format(
                robot_id=robot, x=0, y=0, rotation=0, color=team1GoalieColor
            )
        elif robot in range(2, 20):
            robotsXML += ROBOT_SAMPLE.format(
                robot_id=robot, x=0, y=0, rotation=0, color=team1Color
            )
        elif robot == 21:
            robotsXML += ROBOT_SAMPLE.format(
                robot_id=robot, x=0, y=0, rotation=0, color=team2GoalieColor
            )
        elif robot in range(20, 40):
            robotsXML += ROBOT_SAMPLE.format(
                robot_id=robot, x=0, y=0, rotation=0, color=team2Color
            )
        else:
            raise Exception("Invalid robot number")
    dummyRobotsXML = ""
    for robot in dummyRobots:
        if robot == 1:
            dummyRobotsXML += DUMMY_ROBOT_SAMPLE.format(
                robot_id=robot, x=0, y=0, rotation=0, color=team1GoalieColor
            )
        elif robot in range(2, 20):
            dummyRobotsXML += DUMMY_ROBOT_SAMPLE.format(
                robot_id=robot, x=0, y=0, rotation=0, color=team1Color
            )
        elif robot == 21:
            dummyRobotsXML += DUMMY_ROBOT_SAMPLE.format(
                robot_id=robot, x=0, y=0, rotation=0, color=team2GoalieColor
            )
        elif robot in range(20, 40):
            dummyRobotsXML += DUMMY_ROBOT_SAMPLE.format(
                robot_id=robot, x=0, y=0, rotation=0, color=team2Color
            )
        else:
            raise Exception("Invalid dummy robot number")
    simulationXML = ROS2_SAMPLE.format(
        robots=robotsXML,
        dummy_robots=dummyRobotsXML,
        team1_color=team1Color,
        team1_goalie_color=team1GoalieColor,
        team2_color=team2Color,
        team2_goalie_color=team2GoalieColor,
    )
    return simulationXML


if __name__ == "__main__":
    print(generateRos2([5], [21]))
