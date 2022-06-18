import dask
import numpy as np
import dask.array as da
import dask_ml

def get_random(p):
    x = np.random.random()
    return x < p

def evaluate_cost_and_dists(X, centroids): # (da.Array, np.array) -> float
    distances_matrix = dask_ml.metrics.pairwise_distances(X, centroids) 
    min_distances = da.min(distances_matrix, axis=1) 
    cost = min_distances.sum()
    return cost, min_distances

def get_min_distances(X, centroids):
    distances_matrix = dask_ml.metrics.pairwise_distances(X, centroids)
    min_distances = da.min(distances_matrix, axis=1) 
    return min_distances

def get_closest_centroids_and_dists(X, centroids):
    distances_matrix = dask_ml.metrics.pairwise_distances(X, centroids)
    min_distances = da.min(distances_matrix, axis=1) 
    closest_centroids = da.argmin(distances_matrix, axis=1)
    return closest_centroids, min_distances
 
def oversample(X, distances, l):
    p = l * distances/distances.sum()
    return X[da.random.random(X.shape[0]) < p, :]

def k_means_pp_only_weights(c, counts, k): #explore alternatives
    p = counts/counts.sum()
    idx = np.arange(c.shape[0])
    final_index = np.random.choice(idx, size=k, replace=False, p=p)
    return c[final_index]

def k_means_pp_without_weights(c, weights, k):
    n = c.shape[0]
    idx = np.random.randint(0, n)
    centroids = c[idx, np.newaxis]
    while (centroids.shape[0] < k):
        distances = np.min(skl.metrics.pairwise_distances(c, centroids), axis=1)
        p = distances / distances.sum()
        centroids = np.vstack((centroids, c[np.random.random(c.shape[0]) < p, :]))
    if (centroids.shape[0] > k):
        centroids = np.delete(centroids, np.s_[-1:-1:k-1], axis = 0)
    return centroids

def k_means_pp_weighted(c, weights, k):
    n = c.shape[0]
    idx = np.random.randint(0, n)
    centroids = c[idx, np.newaxis]
    while (centroids.shape[0] < k):
        distances = np.min(skl.metrics.pairwise_distances(c, centroids), axis=1)
        distances = distances * weights
        p = distances / distances.sum()
        centroids = np.vstack((centroids, c[np.random.random(c.shape[0]) < p, :]))
    if (centroids.shape[0] > k):
        centroids = np.delete(centroids, np.s_[-1:-1:k-1], axis = 0)
    return centroids

def k_means_scalable(X, k, l): 
    X = make_da(X)
    n = X.shape[0]
    idx = np.random.randint(0, n)
    centroids = da.compute(X[idx, np.newaxis])[0] #compute() -> np array not stored on nodes.
    initial_cost, distances = da.compute(*evaluate_cost_and_dists(X ,centroids))
    iterations = int(np.round(np.log(initial_cost)))
    for i in range(np.max([iterations, int(k/l)])):
        new_centroids = oversample(X, distances, l).compute()
        centroids = np.vstack((centroids, new_centroids))
        distances = get_min_distances(X, centroids)
    if len(centroids) < k: 
        missing_centroids = k - len(centroids)
        random_index = np.random.choice(a = len(X), size=(1, 3))
        additional_centroids = X[random_index[0]].compute()
        centroids = np.vstack((centroids, additional_centroids))
    closest_centroids, distances = get_closest_centroids_and_dists(X, centroids)
    result = da.unique(closest_centroids, return_counts=True)
    centroid_index, centroid_counts = compute(result)[0]
    centroids_pp = k_means_pp(centroids, centroid_counts, k)
    return centroids_pp #Return initial centroids for Lloyd's algorithm.

def OUR_pairwise_distance(X, centroids):
    
    def min_centroid(y):
        return da.sum(da.square(X - y), axis=1)

    return da.apply_along_axis(min_centroid, 1, centroids).T.compute()
