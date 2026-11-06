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
# [X] SWAP 
# [X] convergence, pass limit 
# [x] final cluster assignment
# [x] choosing K using silhouette
# [ ] print output
# [ ] use actual database

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
        
        # add up all distances 
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

"""
improve selected medoids (until convergence)
"""
def swap(distances, medoids, max_passes=100):
    # TO CONSIDER: proper number of passes and tolerance

    D = np.asarray(distances, dtype=float)

    # check distance matrix
    if D.ndim != 2 or D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be nonempty and square.")
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("Distances must be finite and nonnegative.")
    if not np.allclose(D, D.T) or not np.allclose(np.diag(D), 0):
        raise ValueError("Distance matrix must be symmetric with zero diagonal.")

    n = D.shape[0]

    # check medoids 
    if len(medoids) == 0:
        raise ValueError("At least one medoid is required.")

    for medoid in medoids:
        if isinstance(medoid, (bool, np.bool_)) or not isinstance(medoid, (int, np.integer)):
            raise ValueError("Medoid indices must be integers.")
        if medoid < 0 or medoid >= n:
            raise ValueError("Medoid index is out of range.")

    if len(set(medoids)) != len(medoids):
        raise ValueError("Medoids must not contain duplicates")

    # check stopping conditions
    if isinstance(max_passes, (bool, np.bool_)) or not isinstance(max_passes, (int, np.integer)):
        raise ValueError("max_passes must be an integer.")
    if max_passes < 1:
        raise ValueError("max_passes must be positive.")

    # copy the medoids to make sure original list will be unchanged
    medoids = list(medoids)

    current_cost = total_cost(D, medoids)
    cost_history = [current_cost]

    # DEBUG
    print("\n[SWAP]")
    print("Starting medoids:", medoids)
    print("Starting cost:", current_cost)

    # repeat until no improving swap found OR max number of passes met
    for pass_number in range(1, max_passes + 1):

        # find nearest and second-nearest medoid
        nearest_medoids = []
        nearest_distances = []
        second_distances = []

        for j in range(n):
            nearest = float("inf")
            second = float("inf")
            nearest_medoid = -1

            for medoid in medoids:
                d = D[j][medoid]

                if d < nearest:
                    second = nearest
                    nearest = d
                    nearest_medoid = medoid

                elif d < second:
                    second = d 
            
            nearest_medoids.append(nearest_medoid)
            nearest_distances.append(nearest)
            second_distances.append(second)

        # find pair with smallest T_ih (change in cost when i, h swapped)
        smallest_change = float("inf")
        best_old_medoid = -1
        best_new_medoid = -1

        # i = current medoid
        for old_medoid in medoids:
        
            # h =  non-medoid candidate project
            for candidate in range(n):
                if candidate in medoids:
                    continue

                total_change = 0

                # calculate C_jih (contribution to cost) for every project j
                for j in range(n):
                    # Distance of j to its closest medoid
                    Dj = nearest_distances[j]
                    # Distance of j to its 2nd closest medoid
                    Ej = second_distances[j]
                    # distance of j to h (candidate medoid)
                    distance_to_candidate = D[j][candidate]

                    # Case A: project j belongs to medoid i
                    if nearest_medoids[j] == old_medoid:

                        # Case 2: candidate h is closer to j than its second-nearest medoid
                        if distance_to_candidate < Ej:
                            # C_jih = d(j,h) - d(j,i)
                            contribution = distance_to_candidate - Dj

                        # Case 3: distance of second-nearast medoid to j is closer than or equal to candidate h's
                        else: 
                            # C_jih = Ej - Dj
                            contribution = Ej - Dj

                    # Case b: project j belongs to another medoid
                    else:

                        # Case 1: j stays with current nearest medoid
                        if distance_to_candidate >= Dj:
                            # C_jih = 0
                            contribution = 0

                        # Case 4: h becomes closer to j than its current closest method
                        else: 
                            # Cjih = d(j,h) - Dj
                            contribution = distance_to_candidate - Dj

                    # T_ih = sum of C_jih over j
                    total_change += contribution

                if not np.isfinite(total_change):
                    raise ValueError("ERROR: Swap cost change must be finite.")

                # keep pair with smallest T_ih
                if total_change < smallest_change:
                    smallest_change = total_change
                    best_old_medoid = old_medoid
                    best_new_medoid = candidate

        # Decide whether to carry out swap or stop

        # no possible swap
        if best_old_medoid == -1:
            #DEBUG
            print("No possible swap found")
            print("SWAP stopped")
            break

        # stop if smallest T_ih is positive or zero
        if smallest_change >= 0:
            # DEBUG
            print("\nPass:", pass_number)
            print("Smallest T_ih:", smallest_change)
            print("No swap can decrease the total cost.")
            print("SWAP converged.")
            break

        # smallest T_ih is negative, perform swap
        position = medoids.index(best_old_medoid)

        new_medoids = medoids.copy()
        new_medoids[position] = best_new_medoid

        # calculate the cost after swapping to verify
        new_cost = total_cost(D, new_medoids)

        # check that the swap actually decreased the cost
        if new_cost >= current_cost:
            raise ValueError("Swap did not decrease the total cost.")

        # update the medoids and cost history
        medoids = new_medoids
        cost_history.append(new_cost)

        # DEBUG
        print("\nPass:", pass_number)
        print("Smallest T_ih:", smallest_change)
        print("Replaced medoid:", best_old_medoid)
        print("New medoid:", best_new_medoid)
        print("Previous cost:", current_cost)
        print("New cost:", new_cost)

        current_cost = new_cost

        # go back to SWAP part on the next iteration

    else:
        # when max_passes is reached without breaking
        print("\nMaximum number of passes reached.")

    return medoids, cost_history        

"""
Assign every project to a cluster based on its nearest medoids
"""
def assign_clusters(distances, medoids):
    D = np.asarray(distances, dtype=float)

    # check the distance matrix
    if D.ndim != 2 or D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be nonempty or square.")
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("Distances must be finite and nonnegative")
    if not np.allclose(D, D.T) or not np.allclose(np.diag(D), 0):
        raise ValueError("Distance matrix must be symmetric with zero diagonal.")

    n = D.shape[0]
    medoids = list(medoids)

    # check medoids
    if len(medoids) == 0 or len(medoids) > n:
        raise ValueError("Number of medoids must be between 1 and n")

    for medoid in medoids:
        if medoid < 0 or medoid >= n:
            raise ValueError("Medoid index is out of range.")

    if len(set(medoids)) != len(medoids):
        raise ValueError("Medoids must not contain duplicates.")

    # store each project's medoid label
    labels = np.zeros(n, dtype=int)

    for j in range(n):
        closest_label = 0
        closest_distance = D[j][medoids[0]]

        # compare distances to each select medoid
        for label in range(1, len(medoids)):
            d = D[j][medoids[label]]
            
            # get least distance
            if d < closest_distance:
                closest_distance = d
                closest_label = label

        labels[j] = closest_label
    
    # make sure each medoid represents its own cluster
    for label, medoid in enumerate(medoids):
        labels[medoid] = label

    return labels

"""
Calculate the average silhouette width
"""
def silhouette_score(distances, labels):
    D = np.asarray(distances, dtype=float)
    labels = np.asarray(labels)

    # check distance matrix 
    if D.ndim != 2 or D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be nonempty and square")
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("Distances must be finite and nonnegative.")
    if not np.allclose(D, D.T) or not np.allclose(np.diag(D), 0):
        raise ValueError("Distance matrix must be symmetric with zero diagonal.")

    # check cluster labels
    if labels.ndim != 1 or len(labels) != len(D):
        raise ValueError("There must be one label per project.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Cluster labels must be integers.")

    clusters = np.unique(labels)

    # silhouette is undefined when there's only one cluster
    if len(clusters) < 2:
        raise ValueError("Silhouette is undefined for k=1.")

    # initialize score array
    scores = np.zeros(len(D), dtype=float)

    # calculate silhouette for each project j
    for j in range(len(D)):
        
        # find projects in the same cluster
        same_cluster = np.where(labels == labels[j])[0]

        # silhouette is 0 if j is alone in its cluster (singleton)
        if len(same_cluster) == 1:
            continue

        # a(j) = average distance of j to other projects in its own cluster
        own_distances = []

        for other in same_cluster:
            # exclude j itself
            if other != j:
                own_distances.append(D[j][other])

        a = float(np.mean(own_distances))

        # b(j) = smallest average distance of j to another cluster
        b = float("inf")

        # compare the average distances to all other clusters
        for cluster in clusters:
            if cluster == labels[j]:
                continue  # skip j's own cluster

            # get all projects belonging to this cluster
            other_cluster = np.where(labels == cluster)[0]

            distances_to_cluster = []

            # calculate distances from j to projects in this cluster
            for other in other_cluster:
                distances_to_cluster.append(D[j][other])

            average = float(np.mean(distances_to_cluster))

            # keep the smallest average distance
            if average < b:
                b = average

        # s(j) = (b(j) - a(j)) / max(a(j), b(j))
        # if both are zero, leave the silhouette score at 0
        if max(a, b) > 0:
            scores[j] = (b - a) / max(a, b)                                     

    # check silhouette scores
    if not np.all(np.isfinite(scores)):
        raise ValueError("Silhouette scores must be finite.")
    if np.any(scores < -1 - 1e-10) or np.any(scores > 1 + 1e-10):
        raise ValueError("Silhouette scores must be between -1 and 1.")

    return float(np.mean(scores))

"""
run the entire algo (BUILD, SWAP, Final Cluster Assignments for a given k) 
"""
def run_pam(distances, k):

    # choose starting medoids
    initial_medoids = build(distances, k)

    # improve medoids selection
    final_medoids, cost_history = swap(distances, initial_medoids)

    # assign projects to the final medoids
    labels = assign_clusters(distances, final_medoids)

    # calculate total cost
    cost = total_cost(distances, final_medoids)

    return final_medoids, labels, cost, cost_history

"""
Find the best k using average silhouette width
"""  
def choose_k(distances, k_values):
    D = np.asarray(distances, dtype=float)

    # check distance matrix 
    if D.ndim != 2  or D.shape[0] < 3 or D.shape[0] != D.shape[1]:
        raise ValueError("Choosing k requires atleast 3 projects.")
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("Distances must be finite and nonnegative.")
    if not np.allclose(D, D.T) or not np.allclose(np.diag(D), 0):
        raise ValueError("Distance matrix must be symmetric with zero diagonal.")

    k_values = list(k_values)

    if len(k_values) == 0:
        raise ValueError("At least one k value is required.")

    # k must be at least 2 since silhouette is undefined for k=1
    # exclude k=n since all projects would be in their own clusters
    for k in k_values:
        # check if k is an integer
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
            raise ValueError("k must be an integer.")

        # check if k is within the valid range
        if k < 2 or k >= len(D):
            raise ValueError("Each k must be between 2 and n-1.")

    # make sure k values are not repeated
    # each cluster count should be evaluated only once    
    if len(set(k_values)) != len(k_values):
        raise ValueError("k values must not contain duplicates.")

    # initialize best result dict as output
    best_result = None    

    print("\n=== SILHOUETTE COMPARISON ===")

    # run PAM separately for each k
    for k in k_values:
        medoids, labels, cost, history = run_pam(D, k)

        score = silhouette_score(D, labels)

        print(f"k={k} | silhouette={score:.6f} | "
              f"cost={cost:.6f} | medoids={medoids}")

        # select k with largest average silhouette width
        if best_result is None or score > best_result["score"]:
            best_result = {
                "k": int(k),
                "score": score,
                "medoids": medoids,
                "labels": labels,
                "cost": cost,
                "cost_history": history
            }

    print("Best k:", best_result["k"])

    return best_result

# wala pa
def main():
    pass


if __name__ == "__main__":
    main()
