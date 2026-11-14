import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from kmeans import kmeans
from kmedoids import run_pam
from dbscan import dbscan
from evaluate import distance_matrix, silhouette_score, davies_bouldin_index
from feature_quantifying import prepare_dataset
from config import K, EPS, MIN_PTS, FEATURE_NAMES, SAMPLE_SIZE


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


# generate three panel visualization comparing all algorithms side by side
def plot_comparison(data, all_labels):
    print("\n[status] rendering three panel comparative visualization...")
    colors = plt.get_cmap("tab10") # standard palette for distinct cluster colors
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_titles = [
        f"K-Means (K={K})",
        f"K-Medoids PAM (K={K})",
        f"DBSCAN (eps={EPS}, min_pts={MIN_PTS})"
    ]

    # plot budget versus duration for each method
    for idx in range(3):
        ax = axes[idx]
        labels = all_labels[idx]
        title = plot_titles[idx]

        unique_labels = np.unique(labels)

        # scatter each discovered cluster with distinct color
        for c in unique_labels:
            mask = labels == c

            # handle dbscan noise points specially
            if c == -1:
                ax.scatter(
                    data[mask, 0],
                    data[mask, 1],
                    color="black",
                    marker="x",
                    s=25,
                    alpha=0.7,
                    label="Noise (-1)"
                )
            else:
                ax.scatter(
                    data[mask, 0],
                    data[mask, 1],
                    color=colors(int(c) % 10),
                    marker="o",
                    s=15,
                    alpha=0.6,
                    label="Cluster " + str(c)
                )

        ax.set_xlabel("Budget (normalized)")
        ax.set_ylabel("Duration (normalized)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()

    # save figure in backend outputs folder
    output_dir = os.path.join("backend", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "methods_comparison.png")

    plt.savefig(save_path, dpi=300) # save high resolution plott
    print("[status] saved comparison visualization to:", save_path)
    plt.show()


# export quantitative results to json file
def export_results(results):
    output_dir = os.path.join("backend", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "comparison_results.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("[status] saved quantitative results to:", json_path)


# master workflow executing and comparing algorithms
def main():
    print("=" * 60)
    print("LEAD-AI: MASTER COMPARISON WORKFLOW")
    print("=" * 60)

    # step 1 load dataset
    print("\n[status] fetching and preparing preprocessed dataset...")
    data, rows = prepare_dataset()
    data = np.asarray(data, dtype=float)
    print("[status] dataset loaded successfully with", len(rows), "samples")

    comparison_results = []
    collected_labels = []

    # 1 evaluate kmeans
    print("\n" + "-" * 50)
    print(f"[status] [1/3] initializing k-means clustering (k={K})...")
    print("[status] updating centroids and assigning points...")
    start_time = time.perf_counter()
    km_labels, km_centroids = kmeans(data, K)
    km_runtime = time.perf_counter() - start_time

    print("[status] k-means converged, calculating evaluation metrics...")
    km_sil = silhouette_score(data, km_labels)
    km_dbi = davies_bouldin_index(data, km_labels)

    comparison_results.append({
        "algorithm": "K-Means",
        "num_clusters": K,
        "silhouette_score": float(km_sil),
        "davies_bouldin": float(km_dbi),
        "runtime_seconds": float(km_runtime)
    })
    collected_labels.append(km_labels)
    print("[status] k-means finished in", round(km_runtime, 4), "seconds")

    # 2 evaluate kmedoids pam
    print("\n" + "-" * 50)
    print(f"[status] [2/3] starting k-medoids pam (k={K})...")
    print(f"[status] precomputing pairwise distance matrix ({SAMPLE_SIZE}x{SAMPLE_SIZE})...")
    start_time = time.perf_counter()
    distances = distance_matrix(data)

    print("[status] running pam build phase to select initial medoids...")
    print("[status] executing swap phase until cost convergence (may take a few seconds)...")
    pam_result = run_pam(distances, K)
    kmed_runtime = time.perf_counter() - start_time
    kmed_labels = np.asarray(pam_result[1], dtype=int)

    print("[status] medoids stabilized, calculating evaluation metrics...")
    kmed_sil = silhouette_score(data, kmed_labels)
    kmed_dbi = davies_bouldin_index(data, kmed_labels)

    comparison_results.append({
        "algorithm": "K-Medoids",
        "num_clusters": K,
        "silhouette_score": float(kmed_sil),
        "davies_bouldin": float(kmed_dbi),
        "runtime_seconds": float(kmed_runtime)
    })
    collected_labels.append(kmed_labels)
    print("[status] k-medoids finished in", round(kmed_runtime, 4), "seconds")

    # 3 evaluate dbscan
    print("\n" + "-" * 50)
    print(f"[status] [3/3] starting dbscan density clustering (eps={EPS}, min_pts={MIN_PTS})...")
    print("[status] finding density reachable points and identifying noise...")
    start_time = time.perf_counter()
    db_labels = dbscan(data, eps=EPS, min_pts=MIN_PTS)
    db_runtime = time.perf_counter() - start_time
    db_clusters = len(np.unique(db_labels[db_labels != -1]))

    print("[status] dbscan completed, calculating silhouette and davies bouldin...")
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
    print("[status] dbscan finished in", round(db_runtime, 4), "seconds")

    # display table and save outputs
    print_comparison_table(comparison_results)
    export_results(comparison_results)
    plot_comparison(data, collected_labels)


if __name__ == "__main__":
    main()