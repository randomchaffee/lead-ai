import os
import time
import numpy as np
import matplotlib.pyplot as plt

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


# generate two dimensional scater plots using matplotlib
def plot_kmeans_clusters(data, labels, centroids, k_clusters):
    colors = plt.get_cmap("tab10") # standard color palette for clusters

    # define pairs of features to visualize in two dimensions
    plots_config = [
        (0, 1, "Budget (normalized)", "Duration (normalized)", "K-Means: Budget vs. Duration"),
        (0, 4, "Budget (normalized)", "Contractor Delay Rate (normalized)", "K-Means: Budget vs. Contractor Delay Rate")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # plot each feature pair side by side
    for ax_idx in range(len(plots_config)):
        x_col, y_col, x_label, y_label, title = plots_config[ax_idx]
        ax = axes[ax_idx]

        # plot data points grouped by cluster assignment
        for c in range(k_clusters):
            points = data[labels == c]
            ax.scatter(
                points[:, x_col],
                points[:, y_col],
                color=colors(c),
                alpha=0.6,
                s=20,
                label="Cluster " + str(c)
            )

        # plot calculated centroids as red x markers
        ax.scatter(
            centroids[:, x_col],
            centroids[:, y_col],
            color="red",
            marker="X",
            s=160,
            edgecolors="black",
            linewidths=1.5,
            label="Centroids",
            zorder=5
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title + " (K=" + str(k_clusters) + ")")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # ensure outputs directory exists before saving figure
    output_dir = os.path.join("backend", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "kmeans_clusters.png") # target path to sav chart

    plt.savefig(save_path, dpi=300) # save high resolution plott
    print("saved cluster visualization to:", save_path)
    plt.show()


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

    # render and save scatter plot figures
    plot_kmeans_clusters(data, labels, centroids, k_clusters)

    return data, labels, centroids, rows


if __name__ == "__main__":
    evaluate_kmeans()