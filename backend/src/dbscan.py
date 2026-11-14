import numpy as np

from feature_quantifying import prepare_dataset
from config import EPS, MIN_PTS


UNVISITED = -2
NOISE = -1


def euclidean_distance(first_point, second_point):
    # uclidean distance between two feature vectors
    difference = first_point - second_point
    return float(np.sqrt(np.sum(difference * difference)))


def region_query(data, point_index, eps):
    # indexes of points whose distance is at most eps
    neighbours = []
    point = data[point_index]

    for candidate_index in range(data.shape[0]):
        if euclidean_distance(point, data[candidate_index]) <= eps:
            neighbours.append(candidate_index)

    return neighbours


def _validate_inputs(data, eps, min_pts):
    # DBSCAN parameters and return data as a numeric 2-D array
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("data must be a 2-D array of samples and features")
    if data.shape[0] == 0:
        raise ValueError("data must contain at least one sample")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite numeric values")
    if eps <= 0:
        raise ValueError("eps must be greater than zero")
    if min_pts < 1:
        raise ValueError("min_pts must be at least one")

    return data


def dbscan(data, eps=0.15, min_pts=5):
    #Cluster samples with DBSCAN and return one integer label per sample.

    # euclidean distance is used so the result is comparable with the other methods
    # labels start at zero for discovered clusters
    # points that do not belong to a dense region are labeled -1 as noise
    # min_pts includes the point itself
    
    data = _validate_inputs(data, eps, min_pts)
    sample_count = data.shape[0]
    labels = np.full(sample_count, UNVISITED, dtype=int)
    next_cluster = 0

    for point_index in range(sample_count):
        if labels[point_index] != UNVISITED:
            continue

        neighbours = region_query(data, point_index, eps)
        if len(neighbours) < min_pts:
            labels[point_index] = NOISE
            continue

        labels[point_index] = next_cluster
        seeds = list(neighbours)
        seed_position = 0

        while seed_position < len(seeds):
            seed_index = seeds[seed_position]
            seed_position += 1

            if labels[seed_index] == NOISE:
                labels[seed_index] = next_cluster
            if labels[seed_index] != UNVISITED:
                continue

            labels[seed_index] = next_cluster
            seed_neighbours = region_query(data, seed_index, eps)
            if len(seed_neighbours) >= min_pts:
                for neighbour_index in seed_neighbours:
                    if neighbour_index not in seeds:
                        seeds.append(neighbour_index)

        next_cluster += 1

    return labels


def summarize_clusters(labels):
    # cluster sizes, incl the number of noise points
    labels = np.asarray(labels, dtype=int)
    summary = {}

    for label in labels:
        key = "noise" if label == NOISE else "cluster_" + str(label)
        summary[key] = summary.get(key, 0) + 1

    return summary


def main():
    # run DBSCAN on the shared prepared dataset
    features, rows = prepare_dataset()
    eps = EPS
    min_pts = MIN_PTS
    labels = dbscan(features, eps=eps, min_pts=min_pts)
    
    # terminal
    print("\nDBSCAN results")
    print("samples:", len(rows))
    print("distance metric: Euclidean")
    print("eps:", eps)
    print("min_pts:", min_pts)
    print("cluster sizes:", summarize_clusters(labels))
    print("noise points:", int(np.sum(labels == NOISE)))


if __name__ == "__main__":
    main()
