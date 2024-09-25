"""
Generate the config and scene files we need to run a game
"""

# Default values
TEAM1_NUMBER = "5"
TEAM1_COLOR = "black"
TEAM1_GOALIE_COLOR = "purple"
TEAM2_NUMBER = "70"
TEAM2_COLOR = "red"
TEAM2_GOALIE_COLOR = "blue"

from .ConfigTemplates import *

def generateSceneCon(LogConFileName: str):
    if not LogConFileName.endswith(".con"):
        LogConFileName += ".con"
    return SCENE_CON_TEMPLATE.format(log_con_filename=LogConFileName)

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
            robotsXML += ROBOT_TEMPLATE.format(
                robot_id=robot, x=-idx*25, y=-idx*25, rotation=0, color=team1GoalieColor
            )
        elif robot in range(2, 20):
            robotsXML += ROBOT_TEMPLATE.format(
                robot_id=robot, x=-idx*25, y=-idx*25, rotation=0, color=team1Color
            )
        elif robot == 21:
            robotsXML += ROBOT_TEMPLATE.format(
                robot_id=robot, x=-idx*25, y=-idx*25, rotation=0, color=team2GoalieColor
            )
        elif robot in range(20, 40):
            robotsXML += ROBOT_TEMPLATE.format(
                robot_id=robot, x=-idx*25, y=-idx*25, rotation=0, color=team2Color
            )
        else:
            raise Exception("Invalid robot number")
    dummyRobotsXML = ""
    for idx,robot in enumerate(dummyRobots):
        idx+=1
        if robot == 1:
            dummyRobotsXML += DUMMY_ROBOT_TEMPLATE.format(
                robot_id=robot, x=idx*25, y=idx*25, rotation=0, color=team1GoalieColor
            )
        elif robot in range(2, 20):
            dummyRobotsXML += DUMMY_ROBOT_TEMPLATE.format(
                robot_id=robot, x=idx*25, y=idx*25, rotation=0, color=team1Color
            )
        elif robot == 21:
            dummyRobotsXML += DUMMY_ROBOT_TEMPLATE.format(
                robot_id=robot, x=idx*25, y=idx*25, rotation=0, color=team2GoalieColor
            )
        elif robot in range(20, 40):
            dummyRobotsXML += DUMMY_ROBOT_TEMPLATE.format(
                robot_id=robot, x=idx*25, y=idx*25, rotation=0, color=team2Color
            )
        else:
            raise Exception("Invalid dummy robot number")
    simulationXML = SCENE_ROS2_TEMPLATE.format(
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
    return LOG_DATA_ONLY_CON_TEMPLATE

if __name__ == "__main__":
    print(generateRos2([5], [21]))
