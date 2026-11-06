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
# [ ] final cluster assignment
# [ ] choosing K using silhouette
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

    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, (int, float, np.integer, np.floating)):
        raise ValueError("tol must be a nonnegative finite number.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a nonnegative finite number.")

    # copy the medoids to make sure original list will be unchanged
    medoids = list(medoids)

    current_cost = total_cost(D, medoids)
    cost_history = [current_cost]

    # DEBUG
    print("\n=== SWAP ===")
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

        # recalculate the cost to verify the swap
        new_cost = total_cost(D, new_medoids)
        actual_change = new_cost - current_cost

        # check if the calculated T_ih matches the actual cost change
        if not np.isfinite(new_cost) or not np.isclose(actual_change, smallest_change, rtol=1e-9, atol=tol):
            raise ValueError("ERROR: Swap cost change does not match actual cost.")

        # the new cost must be SMALLER than the prev cost
        if actual_change >= 0:
            raise ValueError("ERROR: Swap did not decrease the total cost.")

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


# wala pa
def main():
    pass

if __name__ == "__main__":
    main()
