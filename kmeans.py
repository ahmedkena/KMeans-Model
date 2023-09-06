import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Read the CSV file and convert the data into a list of points
data = pd.read_csv('CC GENERAL.csv', usecols=range(0,17), index_col="CUST_ID")
data = data.dropna()

points = data.values.tolist()

# Set the number of clusters (k)
k = 3

# Randomly select distinct initial centroids from the dataset
centroids = random.sample(points, k)

# Perform k-means clustering
max_iterations = 100
tolerance = 0.0001

for _ in range(max_iterations):
    clusters = [[] for _ in range(k)]

    # Assign each point to the nearest centroid
    for point in points:
        distances = [math.sqrt(sum([(a - b) ** 2 for a, b in zip(point, centroid)])) for centroid in centroids]
        closest_centroid_idx = distances.index(min(distances))
        clusters[closest_centroid_idx].append(point)

    # Update the centroids by calculating the mean of each cluster
    new_centroids = []
    for cluster in clusters:
        if cluster:  # Skip empty clusters
            centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
            new_centroids.append(centroid)

    # Check if the centroids have converged
    has_converged = True
    for i in range(k):
        if len(new_centroids) <= i:
            has_converged = False
            break
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(centroids[i], new_centroids[i])]))
        if distance > tolerance:
            has_converged = False
            break

    # If the centroids have converged or the number of clusters is less than k, stop the iterations
    if has_converged or len(new_centroids) < k:
        break

    centroids = new_centroids

# Plot the clusters and centroids
colors = ['r', 'g', 'b', 'c', 'm', 'y']  # Add more colors if needed
for i, cluster in enumerate(clusters):
    if cluster:  # Skip empty clusters
        color = colors[i % len(colors)]
        x = [point[0] for point in cluster]
        y = [point[1] for point in cluster]
        plt.scatter(x, y, c=color, label=f'Cluster {i + 1}')

centroids_x = [centroid[0] for centroid in centroids]
centroids_y = [centroid[1] for centroid in centroids]
plt.scatter(centroids_x, centroids_y, c='k', marker='x', label='Centroids')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Clustering Own Code')
plt.legend()
plt.show()

###############################################################################
kmeans = KMeans(n_clusters=3)  # Specify the number of clusters
kmeans.fit(data)  # Fit the model to the data

# Get the cluster labels for each data point
labels = kmeans.labels_

# Add the cluster labels to the DataFrame
data['cluster'] = labels
# Plot the clusters with RGB colors
plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Clustering Library Code')
plt.legend()
plt.show()