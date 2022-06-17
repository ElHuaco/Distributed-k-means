def k_means_scalable(X, k, l): 
    X=make_da(X)
    n = X.shape[0]
    idx = np.random.randint(0, n)
    centroids = da.compute(X[idx, np.newaxis])[0] #since we computed it, it is a numpy array stored here, not on nodes.
    inital_cost = evaluate_cost(X ,centroids).compute()
    iterations = int(np.round(np.log(inital_cost)))
    distances = get_min_distances(X, centroids)
    #init_centroids = da.compute(X[idx, np.newaxis])[0]
    for i in range(np.max([iterations, int(k/l)])):
        print('iteration:', i)
        new_centroids = oversample(X, distances, l).compute()
        print(new_centroids)
        if np.shape(new_centroids) == (0,2):
            continue
        new_distances = get_min_distances(X, new_centroids)
        centroids = np.vstack((centroids, new_centroids))
        distances = da.minimum(new_distances, distances)
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
