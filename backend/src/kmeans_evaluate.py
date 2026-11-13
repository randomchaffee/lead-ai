import time
import numpy as np

from kmeans import kmeans
from evaluate import silhouette_score, davies_bouldin_index
from feature_quantifying import prepare_dataset

# list of feature labels for printing stast
feature_names = [
    "budget",
    "duration_days",
    "contractor_count",
    "contractor_avg_budget",
    "contractor_delay_rate"
]


# display summary of cluster sizes and feature averages
def print_cluster_summary(data, labels, k_clusters):
    print("\ncluster summary:")
    print("--------------------")

    # calculate average values per cluster
    for c in range(k_clusters):
        members = data[labels == c] # projects assigned to cluster c
        count = len(members)

        if count == 0:
            print("cluster", c, "count: 0")
            continue

        means = np.mean(members, axis=0)

        # build string of rounded feature averages
        summary_tokens = []
        for i in range(len(feature_names)):
            entry = feature_names[i] + "=" + str(round(means[i], 4))
            summary_tokens.append(entry)

        print("cluster", c, "count:", count, "means:", ", ".join(summary_tokens))


# run kmeans model and evaluate results
def evaluate_kmeans():
    data, rows = prepare_dataset() # load normalized project matrix
    data = np.asarray(data, dtype=float)

    k_clusters = 4 # four clusters chosen for fair comparison

    # track runtime for kmeans execution
    start_time = time.perf_counter()
    labels, centroids = kmeans(data, k_clusters)
    runtime = time.perf_counter() - start_time

    # calculate cluster quality metrics
    sil = silhouette_score(data, labels)
    dbi = davies_bouldin_index(data, labels)

    # print quantitative output
    print("\nk-means evaluation")
    print("--------------------")
    print("samples:", len(rows))
    print("distance metric: euclidean")
    print("k clusters:", k_clusters)
    print("silhouette score:", round(sil, 4))
    print("davies bouldin index:", round(dbi, 4))
    print("runtime:", round(runtime, 6), "seconds")

    # print cluster averages
    print_cluster_summary(data, labels, k_clusters)

    return data, labels, centroids, rows


if __name__ == "__main__":
    evaluate_kmeans()