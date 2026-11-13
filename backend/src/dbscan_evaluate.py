import time

import numpy as np
import matplotlib.pyplot as plt

from dbscan import dbscan, summarize_clusters
from evaluate import silhouette_score
from feature_quantifying import prepare_dataset

# scatterplot visualization (i used matplotlib)
def plot_clusters(data, labels, eps, min_pts):
    plt.figure(figsize=(9, 6))
    
    for cluster_label in np.unique(labels):
        selected_points = labels == cluster_label
        
        if cluster_label == -1:
            color = "black"
            marker = "x"
            label = "Noise"
            
        else:
            color = None
            marker = "o"
            label = "Cluster " + str(cluster_label)
            
        plt.scatter(
            data[selected_points, 0],
            data[selected_points, 1],
            color=color,
            marker=marker,
            alpha=0.7,
            label=label
        )
        
    plt.xlabel("Budget (normalized)")
    plt.ylabel("Duration (normalized)")
    plt.title(
        "DBSCAN Clusters "
        + "(eps="
        + str(eps)
        + ", min_pts="
        + str(min_pts)
        + ")"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

def main():
    data, rows = prepare_dataset()
    data = np.asarray(data, dtype=float)
    
    eps = 0.15
    min_pts = 5
    
    start_time = time.perf_counter()
    labels = dbscan(data, eps=eps, min_pts=min_pts)
    runtime = time.perf_counter() - start_time
    number_of_cluster = len(
        np.unique(labels[labels != -1])
    )
    
    noise_count = int(np.sum(labels == -1))
    noise_percentage = noise_count / len(labels) * 100
    
    score = silhouette_score(
        data,
        labels,
        ignore_noise=True
    )
    
    print("\nDBSCAN EVALUATION")
    print("--------------------")
    print("samples:", len(rows))
    print("distance metric: Euclidean")
    print("eps:", eps)
    print("min_pts:", min_pts)
    print("number of clusters:", number_of_cluster)
    print("noise points:", noise_count)
    print("noise percentage:", round(noise_percentage, 2), "%")
    print("silhouette score:", round(score, 4))
    print("runtime:", round(runtime, 6), "seconds")
    print("cluster sizes:", summarize_clusters(labels))
    
    # plt scatterplot
    plot_clusters(data, labels, eps, min_pts)
    
if __name__ == "__main__":
    main()