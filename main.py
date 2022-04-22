import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import csv

with open("C:/Users/Admin/Downloads/WhiteHat Python/C - 132 Project/star_with_gravity.csv", "r") as csv_file:
    reader = csv.reader(csv_file)

df = pd.read_csv("star_with_gravity.csv")

mass = df["Mass"].to_list()
radius = df["Radius"].to_list()
dist = df["Distance"].to_list()
gravity = df["Gravity"].to_list()

mass.sort()
radius.sort()
gravity.sort()
plt.plot(radius, mass)
# plt.plot(radius,gravity)

plt.title("Radius & Mass of the Star")
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

plt.plot(mass, gravity)

plt.title("Mass vs Gravity")
plt.xlabel("Mass")
plt.ylabel("Gravity")
plt.show()

plt.scatter(radius, mass)
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

X = df.iloc[:, [3, 4]].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append((kmeans.inertia_))
plt.plot(range(1, 11), wcss)
plt.title("elbow method")
plt.xlabel('Number of clusters')
plt.show()

bools =[]
for d in df.Distance:
    if d<=100:
        bools.append(True)
    else:
        bools.append(False)

is_dist = pd.Series(bools)
is_dist.head()
star_dist=df[is_dist]
star_dist.reset_index(inplace=True, drop=True)
star_dist.head()
star_dist.shape

gravity_bool = []
for g in star_dist.Gravity:
    if g<=350 and g>=150:
        gravity_bool.append(True)
    else :
        gravity_bool.append(False)
is_gravity = pd.Series(gravity_bool)
final_stars = star_dist[is_gravity]
final_stars.head()
final_stars.shape
final_stars.reset_index(inplace=True,drop=True)
final_stars.head()
final_stars.to_csv("filtered_stars.csv")

