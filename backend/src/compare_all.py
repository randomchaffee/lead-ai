import time
import numpy as np

from kmeans import kmeans
from kmedoids import run_pam
from evaluate import distance_matrix, silhouette_score, davies_bouldin_index
from feature_quantifying import prepare_dataset

# feature labels for reference
feature_names = [
    "budget",
    "duration_days",
    "contractor_count",
    "contractor_avg_budget",
    "contractor_delay_rate"
]


# master workflow executing and comparing algorithms
def main():
    print("starting comparison workflow")
    data, rows = prepare_dataset() # load prepared project matrix
    data = np.asarray(data, dtype=float)

    comparison_results = []
    collected_labels = []

    # 1 evaluate kmeans
    print("\n[1/3] executing kmeans...")
    start_time = time.perf_counter()
    km_labels, km_centroids = kmeans(data, 4)
    km_runtime = time.perf_counter() - start_time
    km_sil = silhouette_score(data, km_labels)
    km_dbi = davies_bouldin_index(data, km_labels)

    comparison_results.append({
        "algorithm": "K-Means",
        "num_clusters": 4,
        "silhouette_score": float(km_sil),
        "davies_bouldin": float(km_dbi),
        "runtime_seconds": float(km_runtime)
    })
    collected_labels.append(km_labels)

    print("kmeans completed:")
    print("  clusters: 4")
    print("  silhouette:", round(km_sil, 4))
    print("  davies bouldin:", round(km_dbi, 4))
    print("  runtime:", round(km_runtime, 4), "seconds")

    # 2 evaluate kmedoids pam
    print("\n[2/3] executing kmedoids pam...")
    start_time = time.perf_counter()
    distances = distance_matrix(data)
    pam_result = run_pam(distances, 4)
    kmed_runtime = time.perf_counter() - start_time
    kmed_labels = np.asarray(pam_result[1], dtype=int)
    kmed_sil = silhouette_score(data, kmed_labels)
    kmed_dbi = davies_bouldin_index(data, kmed_labels)

    comparison_results.append({
        "algorithm": "K-Medoids",
        "num_clusters": 4,
        "silhouette_score": float(kmed_sil),
        "davies_bouldin": float(kmed_dbi),
        "runtime_seconds": float(kmed_runtime)
    })
    collected_labels.append(kmed_labels)

    print("kmedoids completed:")
    print("  clusters: 4")
    print("  silhouette:", round(kmed_sil, 4))
    print("  davies bouldin:", round(kmed_dbi, 4))
    print("  runtime:", round(kmed_runtime, 4), "seconds")


if __name__ == "__main__":
    main()