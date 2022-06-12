# Distributed-k-means
The goal of this project is to implement efficiently the **_k-means||_** algorithm in the Dask distributed computing framework, and benchmark the result with some real-world standard datasets made available by sci-kit learn, _v.g._, [RCV1](https://scikit-learn.org/stable/datasets/real_world.html#rcv1-dataset) or [kddcup99](https://scikit-learn.org/stable/datasets/real_world.html#kddcup-99-dataset). 

- ## TODO: Why did we choose Dask VS Spark?
  - https://docs.dask.org/en/stable/spark.html#summary
- ## TODO: How can we download in a distributed manner the dataset?
  - ``dask_ml.datasets`` package if it's defined
  - ``dask.bags`` package: https://docs.dask.org/en/latest/bag-creation.html
  - ``skelearn.datasets`` to a dask dataframe with ``dask.dataframes``: https://docs.dask.org/en/stable/dataframe-create.html
- ## TODO: What optimizations regarding the algorithm have we considered?
    - Once $|x_i-c_j|^2$ is computed for a $(x_i \in X,\text{ } c_j \in C)$ pair, and taken into $\phi_X(C)$, we don't need to compute it again. Moreover, we only need to store it if it's the minimun distance so far.
    - Check if we needed $\mathcal{O}(1)$ instead of $\mathcal{O}(\log\psi)$ iterations in the end
    - Original Dask implementation doesn't follow Step 7 of the **_k-means||_** and just goes to a regular **_k-means++_** with the candidate centroids in Step 8, arguing that the original paper wording remains ambiguous. [See link](https://github.com/dask/dask-ml/blob/main/dask_ml/cluster/k_means.py#L470). We understand two possibilities for these _weighting_ of the candidate centroids mentioned in the paper: either choose the initial centroid _w.p._ $\frac{w_c}{\sum_cw_c}$ or run **_k-means++_** with $\frac{w_c d^2(c, C_f)}{\sum_cw_c d^2(c, C_f)}$ as the distribution.
    - It's possible that the algorithm doesn't return a set of candidate centroids $C$ with $|C| \geq k$. We have two options once this happens: either run additional iterations of the algorithm or choose randomly points until $|C| \geq k$ is satisfied.
- ## TODO: What optimizations regarding Dask have we considered?
  - When did we use ``compute()`` and why
  - Chunks usage: Different arrangements of NumPy arrays will be faster or slower for different algorithms. [See documentation](https://docs.dask.org/en/stable/array-chunks.html). We choose a chunk size of $(0.75\cdot\text{worker RAM, number of features})$. Because we will use repeately the features for calculating distances, but the
- ## TODO: Results of performance benchmarking and comparison with Dask's ``KMeans()``
Below is the Dask **_k-means||_** API, from [dask's examples](https://examples.dask.org/machine-learning/training-on-large-datasets.html?highlight=k%20means).
``` python
km = dask_ml.cluster.KMeans(n_clusters=3, init_max_iter=2, oversampling_factor=10)
km.fit(X)
```
 
