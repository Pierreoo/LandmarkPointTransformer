import numpy as np
from scipy.optimize import least_squares
from sklearn import neighbors
from scipy.sparse.csgraph import shortest_path


# Function to calculate residuals (differences between distances)
def residuals(p, known_points, known_distances):
    graph = neighbors.kneighbors_graph(
        np.vstack((known_points, p.reshape(1, -1))),
        len(known_points) - 1,
        mode='distance',
        include_self=False
    )
    geo_dist = shortest_path(graph, directed=False)
    return geo_dist[-1][:-1] - known_distances


def find_keypoint(known_points, known_distances):
    # Initial guess for the unknown point (use the mean of known points as an initial guess)
    initial_guess = np.mean(known_points, axis=0)
    # Use least_squares to minimize the residuals
    result = least_squares(residuals, initial_guess, args=(known_points, known_distances))
    # The optimized point is the solution
    return result.x
