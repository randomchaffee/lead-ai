import time
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from kmedoids import run_pam
from evaluate import distance_matrix, silhouette_score
from feature_quantifying import prepare_dataset
from config import K


# scatterplot visualization
def plot_clusters(data, labels, medoids, rows, k):
    colors = plt.get_cmap("tab10")

    # features to compare
    plots = [
        (0, 1, "Budget (normalized)",
         "Duration (normalized)", "Budget vs. Duration"),

        (0, 4, "Budget (normalized)",
         "Contractor Delay Rate (normalized)",
         "Budget vs. Contractor Delay Rate")
    ]

    for x_index, y_index, x_label, y_label, title in plots:
        fig, ax = plt.subplots(figsize=(12, max(7, k + 3)))

        # leave space for project info
        fig.subplots_adjust(
            left=0.08,
            right=0.67,
            top=0.90,
            bottom=0.12
        )

        # plot projects based on their clusters
        for cluster_label in np.unique(labels):
            selected_points = labels == cluster_label

            # get the cluster color
            r, g, b, a = colors(int(cluster_label) % 10)

            # make project points a lighter version of the color
            light_color = (
                (3 * r + 1) / 4,
                (3 * g + 1) / 4,
                (3 * b + 1) / 4
            )

            ax.scatter(
                data[selected_points, x_index],
                data[selected_points, y_index],
                color=light_color,
                marker="o",
                s=15,
                alpha=0.8,
                label="Cluster " + str(int(cluster_label) + 1)
            )

        # for medoid points
        for i, index in enumerate(medoids):
            index = int(index)

            # use the stronger color of the medoid's cluster (X mark)
            cluster_label = int(labels[index])
            medoid_color = colors(cluster_label % 10)

            ax.scatter(
                data[index, x_index],
                data[index, y_index],
                color=medoid_color,
                marker="X",
                s=220,
                edgecolors="black",
                linewidths=1,
                zorder=5,
                label="Medoids" if i == 0 else None
            )

            # label medoids with M1, M2, so on
            ax.annotate(
                "M" + str(i + 1),
                (data[index, x_index], data[index, y_index]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=medoid_color,
                zorder=6
            )

        # display actual medoid project details separately
        fig.text(
            0.71, 0.92,
            "SELECTED MEDOID PROJECTS",
            fontsize=11,
            fontweight="bold"
        )

        for i, index in enumerate(medoids):
            project = rows[int(index)]

            # in case project details are incomplete
            contract_id = str(
                project.get("contractId") or "N/A"
            )

            description = str(
                project.get("description") or "No description"
            )

            # in case project details are too long
            short_description = textwrap.shorten(
                description,
                width=65,
                placeholder="..."
            )

            wrapped_description = textwrap.fill(
                short_description,
                width=35
            )

            project_text = (
                "M" + str(i + 1)
                + " | ID: " + contract_id
                + "\n" + wrapped_description
            )

            y_position = 0.85 - i * (0.73 / k)

            # match the project text to its cluster color
            cluster_label = int(labels[int(index)])
            medoid_color = colors(cluster_label % 10)

            fig.text(
                0.71,
                y_position,
                project_text,
                fontsize=9,
                color=medoid_color,
                va="top"
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title + " (K=" + str(k) + ")")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.show()

def main():
    data, rows = prepare_dataset()
    data = np.asarray(data, dtype=float)

    # use constant k from config for fair output comparison
    k = K

    # run K-Medoids using all five normalized features
    start_time = time.perf_counter()

    distances = distance_matrix(data)
    result = run_pam(distances, k)
    runtime = time.perf_counter() - start_time

    medoids = np.asarray(result[0], dtype=int)
    labels = np.asarray(result[1], dtype=int)
    cost = result[2]

    score = silhouette_score(data, labels)

    number_of_clusters = len(np.unique(labels))

    cluster_sizes = {}

    for cluster_label in np.unique(labels):
        cluster_sizes[int(cluster_label) + 1] = int(
            np.sum(labels == cluster_label)
        )

    print("\nK-MEDOIDS EVALUATION")
    print("--------------------")
    print("samples:", len(rows))
    print("distance metric: Euclidean")
    print("k:", k)
    print("number of clusters:", number_of_clusters)
    print("silhouette score:", round(score, 4))
    print("total cost:", round(cost, 4))
    print("runtime:", round(runtime, 6), "seconds")
    print("cluster sizes:", cluster_sizes)

    # display more info about the medoid projects
    print("\nSELECTED MEDOID PROJECTS")
    print("--------------------")

    for cluster_label, index in enumerate(medoids):
        project = rows[int(index)]

        print("\nCluster:", cluster_label)
        print("Contract ID:", project.get("contractId", "N/A"))
        print("Description:", project.get("description", "N/A"))

    # show the two scatter plots
    plot_clusters(data, labels, medoids, rows, k)

if __name__ == "__main__":
    main()
