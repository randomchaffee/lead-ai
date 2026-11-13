import time
import numpy as np

from kmeans import kmeans
from evaluate import silhouette_score, davies_bouldin_index
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


if __name__ == "__main__":
    main()