from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.patches import Polygon as PlotPolygon
from shapely import MultiPolygon, unary_union
from shapely.geometry import Point, Polygon, box

# from .ConvertCfg2Dict import cfg2dict

# Parse the provided data
# fieldData = cfg2dict(
#     Path(__file__).parent.parent
#     / "wistex-system"
#     / "Config/Locations/Default/fieldDimensions.cfg"
# )
fieldData = {
    "xPosOpponentFieldBorder": 5200,
    "xPosOpponentGoal": 5055,
    "xPosOpponentGoalPost": 4525,
    "xPosOpponentGroundLine": 4500,
    "xPosOpponentGoalArea": 3900,
    "xPosOpponentPenaltyMark": 3200,
    "xPosOpponentPenaltyArea": 2850,
    "xPosPenaltyStrikerStartPosition": 2850,
    "xPosHalfWayLine": 0,
    "xPosOwnPenaltyArea": -2850,
    "xPosOwnPenaltyMark": -3200,
    "xPosOwnGoalArea": -3900,
    "xPosOwnGroundLine": -4500,
    "xPosOwnGoalPost": -4525,
    "xPosOwnGoal": -5055,
    "xPosOwnFieldBorder": -5200,
    "yPosLeftFieldBorder": 3700,
    "yPosLeftSideline": 3000,
    "yPosLeftPenaltyArea": 2000,
    "yPosLeftGoalArea": 1100,
    "yPosLeftGoal": 800,
    "yPosRightGoal": -800,
    "yPosRightGoalArea": -1100,
    "yPosRightPenaltyArea": -2000,
    "yPosRightSideline": -3000,
    "yPosRightFieldBorder": -3700,
    "fieldLinesWidth": 50,
    "centerCircleRadius": 750,
    "goalPostRadius": 50,
    "crossBarRadius": 50,
    "goalHeight": 900,
    "penaltyMarkSize": 100,
    "goalFrameLines": [
        {"from": {"x": -4525, "y": 800}, "to": {"x": -5055, "y": 800}},
        {"from": {"x": -5055, "y": 800}, "to": {"x": -5055, "y": -800}},
        {"from": {"x": -4525, "y": -800}, "to": {"x": -5055, "y": -800}},
        {"from": {"x": 4525, "y": 800}, "to": {"x": 5055, "y": 800}},
        {"from": {"x": 5055, "y": 800}, "to": {"x": 5055, "y": -800}},
        {"from": {"x": 4525, "y": -800}, "to": {"x": 5055, "y": -800}},
    ],
    "fieldLines": [
        {"from": {"x": 4500, "y": -3000}, "to": {"x": 4500, "y": 3000}},
        {"from": {"x": 4500, "y": 3000}, "to": {"x": -4500, "y": 3000}},
        {"from": {"x": -4500, "y": 3000}, "to": {"x": -4500, "y": -3000}},
        {"from": {"x": -4500, "y": -3000}, "to": {"x": 4500, "y": -3000}},
        {"from": {"x": 0, "y": 3000}, "to": {"x": 0, "y": -3000}},
        {"from": {"x": -4500, "y": 1100}, "to": {"x": -3900, "y": 1100}},
        {"from": {"x": -3900, "y": 1100}, "to": {"x": -3900, "y": -1100}},
        {"from": {"x": -3900, "y": -1100}, "to": {"x": -4500, "y": -1100}},
        {"from": {"x": 4500, "y": 1100}, "to": {"x": 3900, "y": 1100}},
        {"from": {"x": 3900, "y": 1100}, "to": {"x": 3900, "y": -1100}},
        {"from": {"x": 3900, "y": -1100}, "to": {"x": 4500, "y": -1100}},
        {"from": {"x": -4500, "y": 2000}, "to": {"x": -2850, "y": 2000}},
        {"from": {"x": -2850, "y": 2000}, "to": {"x": -2850, "y": -2000}},
        {"from": {"x": -2850, "y": -2000}, "to": {"x": -4500, "y": -2000}},
        {"from": {"x": 4500, "y": 2000}, "to": {"x": 2850, "y": 2000}},
        {"from": {"x": 2850, "y": 2000}, "to": {"x": 2850, "y": -2000}},
        {"from": {"x": 2850, "y": -2000}, "to": {"x": 4500, "y": -2000}},
        {"from": {"x": 3150, "y": 0}, "to": {"x": 3250, "y": 0}},
        {"from": {"x": 3200, "y": -50}, "to": {"x": 3200, "y": 50}},
        {"from": {"x": -3150, "y": 0}, "to": {"x": -3250, "y": 0}},
        {"from": {"x": -3200, "y": -50}, "to": {"x": -3200, "y": 50}},
        {"from": {"x": -50, "y": 0}, "to": {"x": 50, "y": 0}},
    ],
    "centerCircle": {"center": {"x": 0, "y": 0}, "radius": 750, "numOfSegments": 16},
    "corners": {
        "xCorner": [{"x": 0, "y": 750}, {"x": 0, "y": -750}],
        "tCorner0": [
            {"x": 0, "y": 750},
            {"x": 0, "y": -750},
            {"x": -4500, "y": 2000},
            {"x": -4500, "y": -2000},
            {"x": -4500, "y": 1100},
            {"x": -4500, "y": -1100},
        ],
        "tCorner90": [{"x": 0, "y": 750}, {"x": 0, "y": -750}, {"x": 0, "y": -3000}],
        "tCorner180": [
            {"x": 0, "y": 750},
            {"x": 0, "y": -750},
            {"x": 4500, "y": 2000},
            {"x": 4500, "y": -2000},
            {"x": 4500, "y": 1100},
            {"x": 4500, "y": -1100},
        ],
        "tCorner270": [{"x": 0, "y": 750}, {"x": 0, "y": -750}, {"x": 0, "y": 3000}],
        "lCorner0": [
            {"x": 0, "y": 750},
            {"x": 0, "y": -750},
            {"x": -4500, "y": 2000},
            {"x": -4500, "y": -2000},
            {"x": -4500, "y": 1100},
            {"x": -4500, "y": -1100},
            {"x": 0, "y": -3000},
            {"x": -4500, "y": -3000},
            {"x": 2850, "y": -2000},
            {"x": 3900, "y": -1100},
        ],
        "lCorner90": [
            {"x": 0, "y": 750},
            {"x": 0, "y": -750},
            {"x": 4500, "y": 2000},
            {"x": 4500, "y": -2000},
            {"x": 4500, "y": 1100},
            {"x": 4500, "y": -1100},
            {"x": 0, "y": -3000},
            {"x": 4500, "y": -3000},
            {"x": -2850, "y": -2000},
            {"x": -3900, "y": -1100},
        ],
        "lCorner180": [
            {"x": 0, "y": 750},
            {"x": 0, "y": -750},
            {"x": 4500, "y": 2000},
            {"x": 4500, "y": -2000},
            {"x": 4500, "y": 1100},
            {"x": 4500, "y": -1100},
            {"x": 0, "y": 3000},
            {"x": 4500, "y": 3000},
            {"x": -2850, "y": 2000},
            {"x": -3900, "y": 1100},
        ],
        "lCorner270": [
            {"x": 0, "y": 750},
            {"x": 0, "y": -750},
            {"x": -4500, "y": 2000},
            {"x": -4500, "y": -2000},
            {"x": -4500, "y": 1100},
            {"x": -4500, "y": -1100},
            {"x": 0, "y": 3000},
            {"x": -4500, "y": 3000},
            {"x": 2850, "y": 2000},
            {"x": 3900, "y": 1100},
        ],
    },
}
# Create Shapely objects for the field and important areas


class SoccerFieldPoints:
    centerSpot: Point = Point(0, 0)
    ownPenaltyMark: Point = Point(fieldData["xPosOwnPenaltyMark"], 0)
    opponentPenaltyMark: Point = Point(fieldData["xPosOpponentPenaltyMark"], 0)
    ownLeftGoalPost: Point = Point(
        fieldData["xPosOwnGoalPost"], fieldData["yPosLeftGoal"]
    )
    ownRightGoalPost: Point = Point(
        fieldData["xPosOwnGoalPost"], fieldData["yPosRightGoal"]
    )
    opponentLeftGoalPost: Point = Point(
        fieldData["xPosOpponentGoalPost"], fieldData["yPosLeftGoal"]
    )
    opponentRightGoalPost: Point = Point(
        fieldData["xPosOpponentGoalPost"], fieldData["yPosRightGoal"]
    )


class SoccerFieldAreas:
    fieldBoundary: Polygon = box(
        fieldData["xPosOwnFieldBorder"],
        fieldData["yPosRightFieldBorder"],
        fieldData["xPosOpponentFieldBorder"],
        fieldData["yPosLeftFieldBorder"],
    )
    ownHalf: Polygon = box(
        fieldData["xPosOwnGroundLine"],
        fieldData["yPosRightSideline"],
        fieldData["xPosHalfWayLine"],
        fieldData["yPosLeftSideline"],
    )
    opponentHalf: Polygon = box(
        fieldData["xPosHalfWayLine"],
        fieldData["yPosRightSideline"],
        fieldData["xPosOpponentGroundLine"],
        fieldData["yPosLeftSideline"],
    )
    ownPenaltyArea: Polygon = box(
        fieldData["xPosOwnGroundLine"],
        fieldData["yPosRightPenaltyArea"],
        fieldData["xPosOwnPenaltyArea"],
        fieldData["yPosLeftPenaltyArea"],
    )
    opponentPenaltyArea: Polygon = box(
        fieldData["xPosOpponentPenaltyArea"],
        fieldData["yPosRightPenaltyArea"],
        fieldData["xPosOpponentGroundLine"],
        fieldData["yPosLeftPenaltyArea"],
    )
    ownGoal: Polygon = box(
        fieldData["xPosOwnGoal"],
        fieldData["yPosRightGoal"],
        fieldData["xPosOwnGroundLine"],
        fieldData["yPosLeftGoal"],
    )
    opponentGoal: Polygon = box(
        fieldData["xPosOpponentGoal"],
        fieldData["yPosRightGoal"],
        fieldData["xPosOpponentGroundLine"],
        fieldData["yPosLeftGoal"],
    )
    ownGoalArea: Polygon = box(
        fieldData["xPosOwnGroundLine"],
        fieldData["yPosRightGoalArea"],
        fieldData["xPosOwnGoalArea"],
        fieldData["yPosLeftGoalArea"],
    )
    opponentGoalArea: Polygon = box(
        fieldData["xPosOpponentGoalArea"],
        fieldData["yPosRightGoalArea"],
        fieldData["xPosOpponentGroundLine"],
        fieldData["yPosLeftGoalArea"],
    )
    centerCircle: Polygon = SoccerFieldPoints.centerSpot.buffer(
        fieldData["centerCircleRadius"]
    )
    ownHalfCenterCircle: Polygon = centerCircle.intersection(ownHalf)
    opponentHalfCenterCircle: Polygon = centerCircle.intersection(opponentHalf)


importantPoints = [
    SoccerFieldPoints.centerSpot,
    SoccerFieldPoints.ownPenaltyMark,
    SoccerFieldPoints.opponentPenaltyMark,
    SoccerFieldPoints.ownLeftGoalPost,
    SoccerFieldPoints.ownRightGoalPost,
    SoccerFieldPoints.opponentLeftGoalPost,
    SoccerFieldPoints.opponentRightGoalPost,
]


def Random_Points_in_Bounds(geometry, num_points, random_generator):
    minx, miny, maxx, maxy = geometry.bounds
    x = random_generator.uniform(minx, maxx, num_points)
    y = random_generator.uniform(miny, maxy, num_points)
    return x, y


def generatePoses(geometry, numPoses, rng=None, seed=None):
    # Initialize the random number generator with the provided seed
    if rng is None:
        rng = np.random.default_rng(seed)

    gdf_poly = gpd.GeoDataFrame(index=["myPoly"], geometry=[geometry])

    # Generate more points than needed to account for points outside the geometry
    oversample_factor = 2.0
    x, y = Random_Points_in_Bounds(geometry, int(numPoses * oversample_factor), rng)

    df = pd.DataFrame()
    df["points"] = list(zip(x, y))
    df["points"] = df["points"].apply(Point)
    gdf_points = gpd.GeoDataFrame(df, geometry="points")

    Sjoin = gpd.sjoin(gdf_points, gdf_poly, predicate="within", how="left")
    pnts_in_poly = gdf_points[Sjoin.index_right == "myPoly"]

    # If we didn't get enough points, recursively call the function to get the remaining points
    while len(pnts_in_poly) < numPoses:
        # Generate a new seed for the recursive call
        new_seed = rng.integers(0, 2**32 - 1)
        remaining_points = generatePoses(
            geometry, numPoses - len(pnts_in_poly), seed=new_seed
        )
        pnts_in_poly = pd.concat(
            [pnts_in_poly, gpd.GeoDataFrame(geometry=remaining_points)]
        )

    # Return exactly numPoses points
    result = pnts_in_poly.geometry.iloc[:numPoses].tolist()
    # filter out None values
    result = list(filter(None, result))
    return result


def visualizeField(highlightArea=None, randomPoses=None):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot field boundary
    boundary = PlotPolygon(
        list(SoccerFieldAreas.fieldBoundary.exterior.coords),
        facecolor="lightgreen",
        edgecolor="black",
    )
    ax.add_patch(boundary)

    # Plot areas
    areas = [
        (SoccerFieldAreas.centerCircle, "black", 0.5),
        (SoccerFieldAreas.ownHalf, "blue", 0.1),
        (SoccerFieldAreas.opponentHalf, "red", 0.1),
        (SoccerFieldAreas.ownPenaltyArea, "blue", 0.2),
        (SoccerFieldAreas.opponentPenaltyArea, "red", 0.2),
        (SoccerFieldAreas.ownGoalArea, "blue", 0.3),
        (SoccerFieldAreas.ownGoal, "purple", 0.3),
        (SoccerFieldAreas.opponentGoal, "purple", 0.3),
        (SoccerFieldAreas.opponentGoalArea, "red", 0.3),
    ]

    for area, color, alpha in areas:
        polygon = PlotPolygon(
            list(area.exterior.coords), facecolor=color, alpha=alpha, edgecolor="black"
        )
        ax.add_patch(polygon)

    # Plot important points
    for point in importantPoints:
        circle = Circle(
            (point.x, point.y),
            fieldData["fieldLinesWidth"] / 2,
            facecolor="white",
            edgecolor="black",
        )
        ax.add_patch(circle)

    # Highlight specific area if provided
    if highlightArea:
        if isinstance(highlightArea, Polygon):
            ax.plot(*highlightArea.exterior.xy)
        elif isinstance(highlightArea, MultiPolygon):
            for poly in highlightArea.geoms:
                ax.plot(*poly.exterior.xy)

    # Plot random poses if provided
    if randomPoses:
        x, y = zip(*[(p.x, p.y) for p in randomPoses])
        ax.scatter(x, y, c="black", s=20)

    ax.set_xlim(fieldData["xPosOwnFieldBorder"], fieldData["xPosOpponentFieldBorder"])
    ax.set_ylim(fieldData["yPosRightFieldBorder"], fieldData["yPosLeftFieldBorder"])
    ax.set_aspect("equal")
    ax.set_title("Soccer Field Visualization")
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Random poses in the entire field:")
    fieldPoses = generatePoses(
        unary_union([SoccerFieldAreas.ownPenaltyArea, SoccerFieldAreas.centerCircle]),
        20,
    )
    print(fieldPoses)
    visualizeField(randomPoses=fieldPoses)

    # print("\nRandom poses in own penalty area:")
    # ownPenaltyPoses = generatePoses(SoccerFieldAreas.ownPenaltyArea, 10)
    # print(ownPenaltyPoses)
    # visualizeField(highlightArea=SoccerFieldAreas.ownPenaltyArea, randomPoses=ownPenaltyPoses)

    # You can test other areas similarly
