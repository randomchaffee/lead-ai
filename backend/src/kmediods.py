# k-medoids clustering (PAM, or Partitioning Around Medoids)

# How it works:
#     k-medoids splits the projects into k clusters, where each cluster is represented by a medoid (an actual project that sits at the center of its cluster; minimizes total dissimilarity within its cluster). These clusters can be interpreted as a risk tiers (e.g., low risk, high risk) and used to prioritize audits.

#     The algorithm implemented is based upon the original PAM of Kaufman and Rousseeuw (1990):,

#     a mediod: actual project from the data set that represents its cluster; the one within that cluster w/ the smallest total dissimilarity to other projects in that cluster (p. 68, p. 72)

# The algorithm, step-by-step (unfinalized pa):

#     Main phases (pp. 102-103):
#         a) BUILD: initial clustering via successive selection until k objects have been found
#         b) SWAP: iteratively swaps mediods with non-mediods to further reduce total distance until convergence

#     Step 1: Distance Matrix (Distance = dissimilarity in the paper)
#         D[i][j] = distance between project i and project j
#         NOTE: Distance metric can be either euclidean or manhattan

#     Step 2: Total Cost
#         Each project belongs to its closest mediod (p. 68)
#         Cost of a set of mediods = SUM of the distances from every project to its clsoest mediod (p. 102)

#         NOTE: Minimize the sum instead of the average because the two are mathematically equivalent and the sum is more accurate to compute (p. 102)

#     Step 3: BUILD: choose K starting medoids (pp. 102-103)
#         a) First medoid = the project with the smallest total distance to all other projects (MOST CENTRAL ONE!!!)

#         b) For every project i not yet selected, and every other project j:
#             let Dj = distance from j to its closest selected medoid

#             Cji = max(Dj - d(j,i), 0)

#         c) Total gain of selecting i = sum of Cji over all j

#         d) select project with LARGEST TOTAL GAIN

#         e) repeat b-d until k medoids are selected

#     Step 4: SWAP : improve the medoids (pp. 103-104)
#         a) consider ALL pairs (i, h) where i = selected medoid, h = project not yet selected

#         b) for each pair: find T_ih = change in total cost if i is replaced by h

#             T_ih = sum over j of Cjih, where

#             Contribution to cost, Cjih, is computed based on:
#                 1) Dj (distance from j to its closest medoid)
#                 2) Ej (distnace to its 2nd closest medoid)

#                 - j is closer to another medoid than to i or h:   Cjih = 0
#                 - j belongs to i and d(j,h) < Ej:                 Cjih = d(j,h) - d(j,i)
#                 - j belongs to i and d(j,h) >= Ej:                Cjih = Ej - Dj
#                 - j belongs to another medoid but h becomes closer to j than its current closest medoid                                    Cjih = d(j,h) - Dj

#         CARRY OUT SWAP OR NOT?

#         c) select pair (i, h) with smalled T_ih

#         d) If smallest T_ih is negative, carry out swap and go back to a)

#         e) If smallest T_ih is positive or zero = objective cannot be decreased by any swap -> algorithm stops

#     Step 5: Stopping rule: Convergence (p. 104)
#         Stop when no single swap lowers the total cost

#         NOTE: accept swap only if it lowers the cost by more than 1e-10 (floating point guard), and stops after max iteration passes (TO BE DECIDED, baka default 100) as safety limit

#     Step 6: Final assignment of mediods
#         Given final medoids by this point, every project gets label (0 to k-1) of its closest medoid (which contructs the k clusters, p. 68)

#         INTENDED OUTPUT by then: medoid indices, labels, total cost, cost history

#     Step 7: Choosing K (Silhouette score, Evaluation of k range) (p. 83- 88)

#         Use silhouette, select the k with LARGEST AVERAGE SILHOUETTE WIDTH

#         run PAM for several k, compute average silhouette width of each and compare

#         for each project i:
#             - a(i) = average distance of i to the other objects of its own cluster (p. 83)
#             - b(i) = the smallest, over the other clusters, of the average distance of i to all objects of that cluster (p. 84)

#             s(i) = (b(i) - a(i)) / max(a(i), b(i)), between -1 and 1 (p. 85)
#             ^ value must range from -1 to 1

#             - A project alone in its cluster is given s(i) = 0 (p. 85)

#             - A project in a dataset with only one cluster (k=1) does not have a defined silhouette value.

#           average SILHOUETTE WIDTH for the entire dataset, s(k) = average of s(i) over all projects (p. 86)

#         largest average silhouette width over all test values of (k) = silhouette coefficient (SC) (p. 87)

#         NOTE: for interpretation, refer at table 4 (p. 88), but tl;dr = closer to 1 = well-matched

#     Step 8: Interpration of output (risk tier of each cluster)

#         NOT FINALIZED !!!

#         for each cluster = print size, medoid project (contract id, contractor, budget), and mean of each normalized feature

#         Infer tier from these, for example:

#         large budget w/ high delay = high risk
#         small budget w/ low delay = low risk

#         NOTE: mapping from means to tier names would be OUR OWN INTERPRETATION / CRITERIA










