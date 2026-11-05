import numpy as np
from feature_quantifying import prepare_dataset

# set random seed for reproducibility
np.random.seed(42)

# load preprocessed dataset
# X, rows = prepare_dataset() # X is 1600 by 5 normalized numpy matrix, rows is list of raw dicts

# TODO: remaining PAM implementation 
# [x] distance matrix 
# [X] total cost
# [X] BUILD 
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
        raise ValueError("Points must be nonempty 1-D arrays of equal length.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Point coordinates must be finite.")

    difference = a - b
    return float(np.sqrt(np.sum(difference * difference)))

"""
construct the distance matrix D
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
        raise ValueError("Computed distances must be finite.")

    return distances

"""
add each project's distance to its closest medoid
"""
def total_cost(distances, medoids):
    D = np.asarray(distances, dtype=float)

    # check distance matrix 
    if D.ndim != 2 or D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix msut be nonempty and square.")
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("Distances must be finite or nonnegative.")
    
    n = D.shape[0]

    # check medoid indices
    indices = np.asarray(medoids)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("At least one medoid index is required.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Medoid indices must be integers.")
    if np.any(indices < 0) or np.any(indices >= D.shape[0]):
        raise ValueError("Medoid index out of range.")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("Medoid indices must not contain duplicates.")

    # get distance of each project to its nearest medoid
    cost = 0

    for i in range(n):
        # find the nearest medoid for each project
        nearest = min(D[i][medoid] for medoid in medoids)
        # add distance to total cost
        cost += nearest
    
    return float(cost)

"""
BUILD phase: select initial k medoids
"""
def build(distances, k):

    D = np.asarray(distances, dtype=float)

    # check distance matrix
    if D.ndim != 2 or D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be nonempty and square.")
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("Distances must be finite and nonnegative.")

    n = D.shape[0]

    # check k
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or not 1 <= k <= n:
        raise ValueError("k must be an integer between 1 and n_projects.")    

    medoids = []

    # choose first medoid:
    # project w/ smallest total distance to all other projects
    best_cost = float("inf")
    first_medoid = -1
    
    for i in range(n):
        cost = np.sum(D[i])

        if cost < best_cost:
            best_cost = cost
            first_medoid = i

    if first_medoid == -1 or not np.isfinite(best_cost):
        raise ValueError("ERROR: Could not find a valid first medoid.")

    medoids.append(first_medoid)

    # DEBUG
    # print("First medoid:", first_medoid)
    # print("Total cost:", total_cost(D, medoids)) 

    # choose remaining k-1 medoids
    while len(medoids) < k:
        best_gain = -1
        best_candidate = -1

        # check every project that is not yet a medoid
        for candidate in range(n):
            if candidate in medoids:
                continue

            total_gain = 0

            # calculate gain if this proj becomes a medoid
            for j in range(n):

                # D_j = distance from project j to nearest curr medoid
                nearest = min(D[j][medoid] for medoid in medoids)

                # C_ji = max(D_j - d(j, i), 0)
                gain = nearest - D[j][candidate]

                # only postive gains
                if gain > 0:
                    total_gain += gain
            
            # choose candidate w/ largest total gain
            if total_gain > best_gain:
                best_gain = total_gain
                best_candidate = candidate

        # check candidate validity
        if best_candidate == -1 or not np.isfinite(best_gain):
            raise ValueError("Could not find a valid next medoid.")

        medoids.append(best_candidate)

        # DEBUG
        # print("Added medoid:", best_candidate)
        # print("Gain:", best_gain)
        # print("Total cost:", total_cost(D, medoids))    

    return medoids    

# wala pa
def main():
    pass

if __name__ == "__main__":
    main()
  