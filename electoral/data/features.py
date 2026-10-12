import geopandas as gpd
import pandas as pd
import libpysal


def generate_spatial_features(data: gpd.GeoDataFrame, config: dict) -> gpd.GeoDataFrame:
    """
    The goal of this file is to perform feature engineering carefully to account for spatial relations.
    """


    source = config["data"]["geopackage_path"]
    block_size = config["features"]["block_size_meters"]
    dict_cities = config["features"]["reference_cities"]
    crs = config["data"]["crs"]



    ## FEATURE ENGINEER ##
    #Centroids
    data["CENTROID_X"] = data.geometry.centroid.x
    data["CENTROID_Y"] = data.geometry.centroid.y

    cities_list = [
        {"City": city, "Latitude": coords["lat"], "Longitude": coords["lon"]}
        for city, coords in dict_cities.items()
    ]

    df_cities = pd.DataFrame(cities_list)

    cities = gpd.GeoDataFrame(
        df_cities,
        geometry=gpd.points_from_xy(df_cities["Longitude"], df_cities["Latitude"]),
        crs="EPSG:4326"
    )
    cities = cities.to_crs(crs)

    data["DIST_TO_CITY"] = data.geometry.apply(lambda geom: cities.distance(geom).min())

    # Queen
    wq = libpysal.weights.Queen.from_dataframe(data)
    wq.transform = "r" # Row-standardize

    # LAG
    data["LAG_INDICE_COMPOSITO"] = libpysal.weights.lag_spatial(wq, data["INDICE_COMPOSITO"])


    ## DATA PREPARATION ##


    data["BLOCK_X"] = (data["CENTROID_X"] // block_size).astype(int)
    data["BLOCK_Y"] = (data["CENTROID_Y"] // block_size).astype(int)

    data["BLOCK_ID"] = (
        data["BLOCK_X"].astype(str)
        + "_"
        + data["BLOCK_Y"].astype(str)
    )

    return data