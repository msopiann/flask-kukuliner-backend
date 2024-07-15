from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

import math
import time

from dotenv import load_dotenv
import requests

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["DEBUG"] = os.environ.get("FLASK_DEBUG")

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_DATABASE = os.getenv("DB_DATABASE")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_DATABASE}"

DISTANCEMATRIX_API_KEY = os.getenv("DISTANCEMATRIX_API_KEY")


# Helper function to create a database connection
def get_db_connection():
    try:
        engine = create_engine(DATABASE_URL)
        connection = engine.connect()
        return connection
    except SQLAlchemyError as e:
        print(f"Error connecting to the database: {e}")
        return None


# Helper function to get data from the database
def get_data_from_db():
    try:
        connection = get_db_connection()
        if connection is None:
            return None
        query = "SELECT * FROM listKuliner"
        df = pd.read_sql(query, connection)
        connection.close()
        return df
    except SQLAlchemyError as e:
        print(f"Error executing query: {e}")
        return None


# Services
def get_distance_with_distancematrix_ai(latitude1, longitude1, latitude2, longitude2):
    """Menggunakan DistanceMatrix.AI untuk menghitung jarak antara dua titik."""
    try:
        url = f"https://api.distancematrix.ai/maps/api/distancematrix/json?origins={latitude1},{longitude1}&destinations={latitude2},{longitude2}&key={DISTANCEMATRIX_API_KEY}"
        response = requests.get(url)
        data = response.json()

        if data["rows"][0]["elements"][0]["status"] == "ZERO_RESULTS":
            return None

        distance = data["rows"][0]["elements"][0]["distance"]["value"] / 1000
        return distance
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return None


def get_food_recommendations(
    latitude, longitude, restaurants_data, k=10, max_distance=20
):
    """Get food recommendations based on proximity."""
    try:
        recommended_restaurants = []

        for idx, row in restaurants_data.iterrows():
            restaurant_lat = row["lat"]
            restaurant_lon = row["lon"]

            start_time = time.time()
            distance = get_distance_with_distancematrix_ai(
                latitude, longitude, restaurant_lat, restaurant_lon
            )

            # Check if the request took too long or failed
            if (
                distance is None or (time.time() - start_time) > 5
            ):  # Adjust time limit as needed
                distance = haversine(
                    latitude, longitude, restaurant_lat, restaurant_lon
                )
                method = "Haversine"
            else:
                method = "Distance Matrix"

            if distance is not None and distance <= max_distance:
                restaurant_info = {
                    "id": int(row["id"]),
                    "nama": row["nama"],
                    "description": row["description"],
                    "photoUrl": row["photoUrl"],
                    "estimatePrice": row["estimatePrice"],
                    "lat": restaurant_lat,
                    "lon": restaurant_lon,
                    "distance": f"{distance} | {method}",
                }
                recommended_restaurants.append(restaurant_info)

        recommended_restaurants = sorted(
            recommended_restaurants, key=lambda x: x["distance"]
        )[:k]
        return recommended_restaurants
    except Exception as e:
        print(f"Error in get_food_recommendations: {e}")
        return []


def search_restaurants_by_name(
    names, restaurants_data, latitude=None, longitude=None, limit=10, max_distance=20
):
    """Mencari restoran berdasarkan nama dan mengembalikan hasilnya beserta jaraknya jika lokasi diberikan."""
    results = []

    for name in names:
        filtered_restaurants = restaurants_data[
            restaurants_data["nama"].str.contains(name, case=False, na=False)
        ]
        if latitude is not None and longitude is not None:
            for idx, row in filtered_restaurants.iterrows():
                restaurant_lat = row["lat"]
                restaurant_lon = row["lon"]
                distance = get_distance_with_distancematrix_ai(
                    latitude, longitude, restaurant_lat, restaurant_lon
                )
                if distance is not None and distance <= max_distance:
                    results.append(
                        {
                            "id": row["id"],
                            "nama": row["nama"],
                            "description": row["description"],
                            "photoUrl": row["photoUrl"],
                            "estimatePrice": row["estimatePrice"],
                            "latitude": restaurant_lat,
                            "longitude": restaurant_lon,
                            "distance": distance,
                        }
                    )
        else:
            results.extend(
                filtered_restaurants[
                    [
                        "id",
                        "nama",
                        "description",
                        "photoUrl",
                        "estimatePrice",
                        "lat",
                        "lon",
                    ]
                ].to_dict(orient="records")
            )

    if latitude is not None and longitude is not None:
        results = sorted(results, key=lambda x: x["distance"])[:limit]
    else:
        results = results[:limit]

    return results


def get_restaurant_by_id(id, restaurants_data):
    """Mendapatkan data restoran berdasarkan id."""
    restaurant = restaurants_data[restaurants_data["id"] == str(id)].to_dict(
        orient="records"
    )
    if restaurant:
        return restaurant[0]
    else:
        return None


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Kukuliner Endpoint API using Flask"})


@app.route("/api/culinary", methods=["GET"])
def get_all_restaurants():
    restaurants_data = get_data_from_db()
    if restaurants_data is None:
        return jsonify({"error": "Error retrieving data from database"}), 500

    restaurants_list = restaurants_data.to_dict(orient="records")
    return jsonify({"listKuliner": restaurants_list})


@app.route("/api/culinary/search", methods=["GET"])
def search_restaurant():
    names = request.args.getlist("name")
    latitude = request.args.get("lat", type=float, default=None)
    longitude = request.args.get("lon", type=float, default=None)

    if not names:
        return jsonify({"error": "Name parameter is required"}), 400

    restaurants_data = get_data_from_db()
    if restaurants_data is None:
        return jsonify({"error": "Error retrieving data from database"}), 500

    search_results = search_restaurants_by_name(
        names, restaurants_data, latitude, longitude
    )
    return jsonify({"listKuliner": search_results})


@app.route("/api/culinary/<int:id>", methods=["GET"])
def get_single_restaurant(id):
    restaurants_data = get_data_from_db()
    if restaurants_data is None:
        return jsonify({"error": "Error retrieving data from database"}), 500

    restaurant = get_restaurant_by_id(id, restaurants_data)
    if restaurant:
        desired_order = [
            "id",
            "nama",
            "description",
            "photoUrl",
            "estimatePrice",
            "lat",
            "lon",
        ]
        sorted_restaurant = {key: restaurant[key] for key in desired_order}
        return jsonify({"listKuliner": sorted_restaurant})
    else:
        return jsonify({"error": "Restaurant not found"}), 404


@app.route("/api/culinary/recommendations", methods=["GET"])
def recommend_food():
    latitude = request.args.get("lat")
    longitude = request.args.get("lon")

    if latitude is None or longitude is None:
        return (
            jsonify({"error": "Latitude and longitude parameters are required."}),
            400,
        )

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError:
        return jsonify({"error": "Latitude and longitude must be float values."}), 400

    restaurants_data = get_data_from_db()
    if restaurants_data is None:
        return jsonify({"error": "Error retrieving data from database"}), 500

    recommended_food = get_food_recommendations(latitude, longitude, restaurants_data)
    return jsonify({"listKuliner": recommended_food})


# BONUS HAVERSINE FORMULA
# Define Earth radius (in kilometers)
EARTH_RADIUS = 6371


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points on a sphere using the Haversine formula.

    Args:
        lat1: Latitude of the first point in degrees.
        lon1: Longitude of the first point in degrees.
        lat2: Latitude of the second point in degrees.
        lon2: Longitude of the second point in degrees.

    Returns:
        The distance between the two points in kilometers.
    """
    # Convert latitudes and longitudes to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(lat1_rad) * math.cos(
        lat2_rad
    ) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = EARTH_RADIUS * c

    return distance


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
