import dask
import numpy as np
import dask.array as da
import dask_ml

def get_random(p):
    x = np.random.random()
    return x < p

def evaluate_cost(X, centroids): # (da.Array, np.array) -> float
    distances_matrix = dask_ml.metrics.pairwise_distances(X, centroids) 
    tot = distances_matrix.min(axis=1).sum()
    return tot

def get_min_distances(X, centroids):
    distances_matrix = dask_ml.metrics.pairwise_distances(X, centroids)
    min_distances= da.min(distances_matrix, axis=1) 
    return min_distances 

def closest_c(X, centroids):
    distances_matrix = dask_ml.metrics.pairwise_distances(X, centroids)
    closest_centroid= da.argmin(distances_matrix, axis=1)
    return closest_centroid
    
def update(X, distances, l):
    p = l * distances/distances.sum()
    mask = da.map_blocks(get_random ,p)
    return X[mask,:]

def k_means_pp(centroids, counts, k): #explore alternatives
    probs= counts/counts.sum()
    tot_init= len(centroids)
    centroid_index=np.arange(len(centroids))
    final_index= np.random.choice(centroid_index, size=k, replace=False, p=probs)
    return centroids[final_index]

def k_means_scalable(X, k, l): 
    X=make_da(X)
    n = X.shape[0]
    idx = np.random.randint(0, n)
    centroids = da.compute(X[idx, np.newaxis])[0] #since we computed it, it is a numpy array stored here, not on nodes.
    inital_cost = evaluate_cost(X ,centroids).compute()
    iterations = int(np.round(np.log(inital_cost)))
    for i in range(np.max([iterations, int(k/l)])):
        distances = get_min_distances(X, centroids)
        new_centroids = update(X, distances, l).compute()
        centroids = np.vstack((centroids, new_centroids))
    if len(centroids) < k : #this raises an error it need to be written again
        missing_centroids = k - len(centroids)
        random_index = np.random.choice(a = len(X), size=(1, 3))
        additional_centroids = X[random_index[0]].compute()
        centroids = np.vstack((centroids, additional_centroids))#fix this
    final_distances  = get_min_distances(X, centroids)
    final_closest_centroid = closest_c(X, centroids)
    result= da.unique(final_closest_centroid, return_counts=True)
    centroid_index, centroid_counts= compute(result)[0]
    centroids_pp = k_means_pp(centroids, centroid_counts, k)
    return centroids_pp #this are the initial centroids for the Lloyd's algorithm.


def OUR_pairwise_distance(X, centroids):
    
    def min_centroid(y):
        return np.square(x - y).sum(axis=1)

    return np.array(list(map(min_centroid,y))).T
