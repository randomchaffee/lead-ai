import numpy as np
from feature_quantifying import prepare_dataset

# set random seed for reproducibility
np.random.seed(42)

# number of clusters (same as the number of clusters DBSCAN found)
K = 4

# labels for the five features from feature_quantifying.py
FEATURE_NAMES = [
    "budget",
    "duration_days",
    "contractor_count",
    "contractor_avg_budget",
    "contractor_delay_rate"
]


def distance_matrix(data):
    # Euclidean distance between every pair of projects (n by n).
    points = np.asarray(data, dtype=float)

    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("Data must be a nonempty 2-D array of project features.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Project features must be finite.")

    difference = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(difference * difference, axis=2))


def assign_clusters(X, centroids):
    # Label each project with its nearest centroid (ties go to the lowest index).
    # squared Euclidean distance from every project to every centroid -> (n, k)
    dists = np.zeros((X.shape[0], centroids.shape[0]))

    for c in range(centroids.shape[0]):
        dists[:, c] = np.sum((X - centroids[c]) ** 2, axis=1)

    return np.argmin(dists, axis=1)


def update_centroids(X, labels, centroids):
    # Move each centroid to the coordinate-wise mean of its members.
    new_centroids = centroids.copy()

    for c in range(centroids.shape[0]):
        members = X[labels == c]
        # an empty cluster keeps its previous centroid
        if len(members) > 0:
            new_centroids[c] = members.mean(axis=0)

    return new_centroids


def kmeans(X, k, max_iter=300):
    # Run K-means and return (labels, centroids).
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("Data must be a nonempty 2-D array.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Data must contain only finite values.")
    if not 1 <= k <= X.shape[0]:
        raise ValueError("k must be between 1 and the number of projects.")

    # k random observations become the starting centroids
    start = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X[start].copy()
    labels = None

    for iteration in range(1, max_iter + 1):
        print("updating centroid [iteration]:", iteration)

        # assign every project to its nearest centroid
        new_labels = assign_clusters(X, centroids)

        # stop when the assignments no longer change
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # recompute the centroids
        centroids = update_centroids(X, labels, centroids)
    else:
        print("\nMaximum number of iterations reached.")

    return labels, centroids


def silhouette_score(distances, labels):
    # Average silhouette width, computed from the Euclidean distance matrix.
    D = np.asarray(distances, dtype=float)
    labels = np.asarray(labels)

    if D.ndim != 2 or D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be nonempty and square.")
    if labels.ndim != 1 or len(labels) != len(D):
        raise ValueError("There must be one label per project.")

    clusters = np.unique(labels)
    if len(clusters) < 2:
        raise ValueError("Silhouette is undefined for fewer than 2 clusters.")

    n = len(D)

    # mean_dist[j, c] = average distance from project j to all members of cluster c
    mean_dist = np.zeros((n, len(clusters)))
    for position, cluster in enumerate(clusters):
        mask = labels == cluster
        mean_dist[:, position] = D[:, mask].sum(axis=1) / mask.sum()

    own = np.searchsorted(clusters, labels)
    sizes = np.array([np.sum(labels == label) for label in labels])

    # a(j): average distance to the other members of its own cluster
    # (the sum includes the zero distance to itself, so divide by size - 1)
    a = np.zeros(n)
    has_peers = sizes > 1
    a[has_peers] = mean_dist[np.arange(n), own][has_peers] * sizes[has_peers] / (sizes[has_peers] - 1)

    # b(j): smallest average distance to any other cluster
    other = mean_dist.copy()
    other[np.arange(n), own] = np.inf
    b = other.min(axis=1)

    # s(j) = (b - a) / max(a, b); singleton clusters score 0
    scores = np.zeros(n)
    denominator = np.maximum(a, b)
    valid = has_peers & (denominator > 0)
    scores[valid] = (b[valid] - a[valid]) / denominator[valid]

    return float(np.mean(scores))


def print_cluster_summary(X, labels, k):
    # Print one row per cluster: size and average normalized features.
    print("\nCluster    # Projects    Average normalized features")
    print("-" * 90)

    for c in range(k):
        members = X[labels == c]

        if len(members) == 0:
            print(f"{c:<10} {0:<13} (empty cluster)")
            continue

        means = members.mean(axis=0)
        summary = ", ".join(f"{name}={value:.2f}" for name, value in zip(FEATURE_NAMES, means))
        print(f"{c:<10} {len(members):<13} {summary}")


def main():
    # load actual project data using the existing preprocessing code
    X, rows = prepare_dataset()
    X = np.asarray(X, dtype=float)

    # check project data
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Expected five project features.")
    if len(X) != len(rows):
        raise ValueError("Each feature row must match one project record.")
    if K < 2 or K >= len(X):
        raise ValueError("K must be between 2 and n-1.")

    print("\nK-MEANS RESULTS")
    print("Total # of projects:", len(rows))
    print("Features per project:", X.shape[1])
    print(f"\nFinding k={K} clusters...\n")

    labels, centroids = kmeans(X, K)

    # silhouette uses true Euclidean distances, same as the K-medoids script
    score = silhouette_score(distance_matrix(X), labels)
    print("\nAverage silhouette:", round(score, 4))

    print_cluster_summary(X, labels, K)


if __name__ == "__main__":
    main()