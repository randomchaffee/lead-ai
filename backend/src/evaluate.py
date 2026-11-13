import numpy as np

# evaluation metrics

def distance_matrix(data):
    """
    creates a euclidian distance matrix.
    use this to spare yourself from writing boilerplate code.
    """
    data = np.asarray(data, dtype=float)
    
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("data needs to be a nonempty 2D array")
    
    diff = data[:, None, :] - data[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

# calculate the average silhouette score
def silhouette_score(data, labels, ignore_noise=False):
    data = np.asarray(data, dtype=float)
    labels = np.asarray(labels, dtype=int)
    
    if len(data) != len(labels):
        raise ValueError("data and labels need to be the same length")
    
    if ignore_noise:
        keep = labels != -1
        data = data[keep]
        labels = labels[keep]
    
    clusters = np.unique(labels)
    
    if len(clusters) < 2:
        raise ValueError("silhouette requires at least 2 clusters")
    
    distances = distance_matrix(data)
    scores = [] # sil score storage
    
    for point_idx in range(len(data)):
        own_cluster = labels[point_idx]
        own_points = labels == own_cluster
        
        if np.sum(own_points) <= 1:
            scores.append(0.0)
            continue
        
        own_distances = distances[point_idx, own_points]
        a = np.sum(own_distances) / (np.sum(own_points) - 1)
        
        other_cluster_averages = []
        
        for cluster in clusters:
            if cluster == own_cluster:
                continue
            
            other_points = labels == cluster
            average_distance = np.mean(
                distances[point_idx, other_points]
            )
            other_cluster_averages.append(average_distance)
            
        b = min(other_cluster_averages)
        
        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))
    
    return float(np.mean(scores))

def davies_bouldin_index(data, labels, ignore_noise=False):
    data = np.asarray(data, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if data.ndim != 2 or len(data) == 0:
        raise ValueError("data must be a nonempty 2D array")

    if len(data) != len(labels):
        raise ValueError("data and labels need to be the same length")

    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values")

    if ignore_noise:
        keep = labels != -1
        data = data[keep]
        labels = labels[keep]

    clusters = np.unique(labels)

    if len(clusters) < 2:
        raise ValueError(
            "Davies-Bouldin Index requires at least 2 clusters"
        )

    centroids = []
    scatters = []

    for cluster in clusters:
        members = data[labels == cluster]
        centroid = np.mean(members, axis=0)

        distances = np.sqrt(
            np.sum((members - centroid) ** 2, axis=1)
        )

        centroids.append(centroid)
        scatters.append(np.mean(distances))

    centroids = np.asarray(centroids)
    scatters = np.asarray(scatters)

    centroid_distances = distance_matrix(centroids)
    ratios = []

    for i in range(len(clusters)):
        cluster_ratios = []

        for j in range(len(clusters)):
            if i == j:
                continue

            distance = centroid_distances[i, j]
            numerator = scatters[i] + scatters[j]

            if distance == 0:
                ratio = np.inf if numerator > 0 else 0.0
            else:
                ratio = numerator / distance

            cluster_ratios.append(ratio)

        ratios.append(max(cluster_ratios))

    return float(np.mean(ratios))