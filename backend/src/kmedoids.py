import numpy as np
from feature_quantifying import prepare_dataset

# set random seed for reproducibility
np.random.seed(42)

# load preprocessed dataset
# X, rows = prepare_dataset() # X is 1600 by 5 normalized numpy matrix, rows is list of raw dicts

# TODO: remaining PAM implementation 
# [x] distance matrix 
# [ ] total cost
# [ ] BUILD 
# [ ] SWAP 
# [ ] convergence, pass limit
# [ ] final cluster assignment
# [ ] choosing K using silhouette
# [ ] print output

"""
calculate the distance between two projects
"""
def euclidean_distance(point_a, point_b):
    # return the Euclidean distance between two one-dimensional points
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)

    # each project must have the same nonempty feature set
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("ERROR: Points must be nonempty 1-D arrays of equal length.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("ERROR: Point coordinates must be finite.")

    difference = a - b
    return float(np.sqrt(np.sum(difference * difference)))

"""
construct distance matrix D
"""
def distance_matrix(data):
    points = np.asarray(data, dtype=float)

    # require a valid project-by-feature matrix
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("Data must be a nonempty 2-D array of project features.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Project features must be finite.")

    # number of rows in arr points
    n = points.shape[0]

    # initialize distance matrix
    distances = np.zeros((n, n), dtype=float)

    # compute every unique pair of project distances
    # Euclidean distance is symmetric;
    # lower triangle mirrors upper triangle,   
    # diagonal remains 0
    for i in range(n):
        for j in range(n):
            if i < j:
                d = euclidean_distance(points[i], points[j])
                distances[i, j] = d 
                distances[j, i] = d
    
    # TO CONSIDER: paper recommend storing only lower triangle for optimization

    # prevent overflow during distance calculation
    if not np.all(np.isfinite(distances)):
        raise ValueError("ERROR: Computed distances must be finite.")

    return distances

# wala pa
def main():
    pass

if __name__ == "__main__":
    main()
  