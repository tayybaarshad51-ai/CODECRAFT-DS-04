import pandas as pd
import matplotlib.pyplot as plt

# Read the large dataset in chunks
file = "US_Accidents_March23.csv"

hours = []
weather = []
states = []
cities = []

print("Reading accident dataset...")

for chunk in pd.read_csv(
    file,
    usecols=["Start_Time", "Weather_Condition", "State", "City"],
    chunksize=100000,
    low_memory=False
):
    chunk["Start_Time"] = pd.to_datetime(
        chunk["Start_Time"], errors="coerce"
    )

    # Hour of accident
    hours.extend(chunk["Start_Time"].dt.hour.dropna().tolist())

    # Weather
    weather.extend(
        chunk["Weather_Condition"].dropna().tolist()
    )

    # State
    states.extend(
        chunk["State"].dropna().tolist()
    )

    # City
    cities.extend(
        chunk["City"].dropna().tolist()
    )

print("\nUS Traffic Accident Analysis")
print("----------------------------")

# -------------------------
# 1. Time of day
# -------------------------
hour_counts = pd.Series(hours).value_counts().sort_index()

print("\nAccidents by Hour:")
print(hour_counts)

plt.figure(figsize=(10, 5))
hour_counts.plot(kind="bar")
plt.title("Traffic Accidents by Time of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

# -------------------------
# 2. Weather conditions
# -------------------------
weather_counts = pd.Series(weather).value_counts().head(10)

print("\nTop Weather Conditions:")
print(weather_counts)

plt.figure(figsize=(10, 5))
weather_counts.plot(kind="bar")
plt.title("Top 10 Weather Conditions During Accidents")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# -------------------------
# 3. States
# -------------------------
state_counts = pd.Series(states).value_counts().head(10)

print("\nTop 10 States:")
print(state_counts)

plt.figure(figsize=(10, 5))
state_counts.plot(kind="bar")
plt.title("Top 10 States by Accident Count")
plt.xlabel("State")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

# -------------------------
# 4. Accident hotspots
# -------------------------
city_counts = pd.Series(cities).value_counts().head(10)

print("\nTop 10 Accident Hotspots:")
print(city_counts)

plt.figure(figsize=(10, 5))
city_counts.plot(kind="bar")
plt.title("Top 10 Accident Hotspots by City")
plt.xlabel("City")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("\nAnalysis completed successfully!")