import time
import numpy as np

from kmeans import kmeans
from kmedoids import run_pam
from dbscan import dbscan
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


# format and print side by side comaprison table in terminal
def print_comparison_table(results):
    print("\n" + "=" * 80)
    print("LEAD-AI: CLUSTERING METHODS COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<15} | {'Clusters':<10} | {'Silhouette':<12} | {'Davies-Bouldin':<15} | {'Runtime (s)':<12}")
    print("-" * 80)

    for r in results:
        name = r["algorithm"]
        k_val = str(r["num_clusters"])
        sil_str = str(round(r["silhouette_score"], 4))
        dbi_str = str(round(r["davies_bouldin"], 4))
        time_str = str(round(r["runtime_seconds"], 4))
        print(f"{name:<15} | {k_val:<10} | {sil_str:<12} | {dbi_str:<15} | {time_str:<12}")

    print("=" * 80)


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

    # 3 evaluate dbscan
    print("\n[3/3] executing dbscan...")
    start_time = time.perf_counter()
    db_labels = dbscan(data, eps=0.15, min_pts=5)
    db_runtime = time.perf_counter() - start_time
    db_clusters = len(np.unique(db_labels[db_labels != -1]))
    db_sil = silhouette_score(data, db_labels, ignore_noise=True)
    db_dbi = davies_bouldin_index(data, db_labels, ignore_noise=True)
    db_noise_count = int(np.sum(db_labels == -1))

    comparison_results.append({
        "algorithm": "DBSCAN",
        "num_clusters": db_clusters,
        "noise_points": db_noise_count,
        "silhouette_score": float(db_sil),
        "davies_bouldin": float(db_dbi),
        "runtime_seconds": float(db_runtime)
    })
    collected_labels.append(db_labels)

    # display formatted terminal table
    print_comparison_table(comparison_results)


if __name__ == "__main__":
    main()