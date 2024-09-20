import json
import numpy as np
from shapely import MultiPolygon, unary_union
from shapely.geometry import Polygon, Point, box
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as PlotPolygon
import geopandas as gpd
import pandas as pd
from .ConvertCfg2Dict import cfg2dict

# Parse the provided data
fieldData = cfg2dict(
    "/home/yuhao2024/Documents/BHumanCodeRelease-origin/Config/Locations/Default/fieldDimensions.cfg"
)
# Create Shapely objects for the field and important areas

fieldBoundary = box(
    fieldData["xPosOwnFieldBorder"],
    fieldData["yPosRightFieldBorder"],
    fieldData["xPosOpponentFieldBorder"],
    fieldData["yPosLeftFieldBorder"],
)

ownHalf = box(
    fieldData["xPosOwnGroundLine"],
    fieldData["yPosRightSideline"],
    fieldData["xPosHalfWayLine"],
    fieldData["yPosLeftSideline"],
)

opponentHalf = box(
    fieldData["xPosHalfWayLine"],
    fieldData["yPosRightSideline"],
    fieldData["xPosOpponentGroundLine"],
    fieldData["yPosLeftSideline"],
)

ownPenaltyArea = box(
    fieldData["xPosOwnGroundLine"],
    fieldData["yPosRightPenaltyArea"],
    fieldData["xPosOwnPenaltyArea"],
    fieldData["yPosLeftPenaltyArea"],
)

opponentPenaltyArea = box(
    fieldData["xPosOpponentPenaltyArea"],
    fieldData["yPosRightPenaltyArea"],
    fieldData["xPosOpponentGroundLine"],
    fieldData["yPosLeftPenaltyArea"],
)

ownGoalArea = box(
    fieldData["xPosOwnGroundLine"],
    fieldData["yPosRightGoalArea"],
    fieldData["xPosOwnGoalArea"],
    fieldData["yPosLeftGoalArea"],
)

opponentGoalArea = box(
    fieldData["xPosOpponentGoalArea"],
    fieldData["yPosRightGoalArea"],
    fieldData["xPosOpponentGroundLine"],
    fieldData["yPosLeftGoalArea"],
)

centerCircle = Point(0, 0).buffer(fieldData["centerCircleRadius"])

# Create important points
centerSpot = Point(0, 0)
ownPenaltyMark = Point(fieldData["xPosOwnPenaltyMark"], 0)
opponentPenaltyMark = Point(fieldData["xPosOpponentPenaltyMark"], 0)
ownLeftGoalPost = Point(fieldData["xPosOwnGoalPost"], fieldData["yPosLeftGoal"])
ownRightGoalPost = Point(fieldData["xPosOwnGoalPost"], fieldData["yPosRightGoal"])
opponentLeftGoalPost = Point(
    fieldData["xPosOpponentGoalPost"], fieldData["yPosLeftGoal"]
)
opponentRightGoalPost = Point(
    fieldData["xPosOpponentGoalPost"], fieldData["yPosRightGoal"]
)

importantPoints = [
    centerSpot,
    ownPenaltyMark,
    opponentPenaltyMark,
    ownLeftGoalPost,
    ownRightGoalPost,
    opponentLeftGoalPost,
    opponentRightGoalPost,
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
        list(fieldBoundary.exterior.coords), facecolor="lightgreen", edgecolor="black"
    )
    ax.add_patch(boundary)

    # Plot areas
    areas = [
        (centerCircle, "black", 0.5),
        (ownHalf, "blue", 0.1),
        (opponentHalf, "red", 0.1),
        (ownPenaltyArea, "blue", 0.2),
        (opponentPenaltyArea, "red", 0.2),
        (ownGoalArea, "blue", 0.3),
        (opponentGoalArea, "red", 0.3),
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
    fieldPoses = generatePoses(unary_union([ownPenaltyArea, centerCircle]), 20)
    print(fieldPoses)
    visualizeField(randomPoses=fieldPoses)

    # print("\nRandom poses in own penalty area:")
    # ownPenaltyPoses = generatePoses(ownPenaltyArea, 10)
    # print(ownPenaltyPoses)
    # visualizeField(highlightArea=ownPenaltyArea, randomPoses=ownPenaltyPoses)

    # You can test other areas similarly
