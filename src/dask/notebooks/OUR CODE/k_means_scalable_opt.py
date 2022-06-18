def k_means_scalable(X, k, l): 
    X = make_da(X)
    n_points, n_features = X.shape
    idx = np.random.randint(0, n)
    centroids = da.compute(X[idx, np.newaxis])[0] #compute() -> np array not stored on nodes.
    inital_cost, distances = evaluate_cost(X ,centroids).compute()
    iterations = int(np.round(np.log(inital_cost)))
    for i in range(np.max([iterations, int(k/l)])):
        new_centroids = oversample(X, distances, l).compute()
        if np.shape(new_centroids) == (0, n_features):
            continue
        new_distances = get_min_distances(X, new_centroids)
        centroids = np.vstack((centroids, new_centroids))
        distances = da.minimum(new_distances, distances)
    if len(centroids) < k:
        missing_centroids = k - len(centroids)
        random_index = np.random.choice(a = len(X), size=(1, 3))
        additional_centroids = X[random_index[0]].compute()
        centroids = np.vstack((centroids, additional_centroids))
    closest_centroids, distances = get_closest_centroids_and_dists(X, centroids)
    result = da.unique(final_closest_centroid, return_counts=True)
    centroid_index, centroid_counts= compute(result)[0]
    centroids_pp = k_means_pp(centroids, centroid_counts, k)
    return centroids_pp #Return initial centroids for Lloyd's algorithm.
