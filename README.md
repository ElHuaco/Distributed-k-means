# Distributed-k-means
The goal of this project is to implement efficiently the **_k-means||_** algorithm in the Dask distributed computing framework, and benchmark the result with some real-world standard datasets made available by sci-kit learn, _v.g._, [RCV1](https://scikit-learn.org/stable/datasets/real_world.html#rcv1-dataset) or [kddcup99](https://scikit-learn.org/stable/datasets/real_world.html#kddcup-99-dataset).
## Implementation
- ### TODO: Downloading the dataset in a distributed manner:
  - ``dask_ml.datasets`` package if it's defined
  - ``dask.bags`` package: https://docs.dask.org/en/latest/bag-creation.html
  - ``skelearn.datasets`` to a dask dataframe with ``dask.dataframes``: https://docs.dask.org/en/stable/dataframe-create.html
- ### TODO: Dask best practices considered:
  - When did we use ``compute()`` and why
  - Chunks usage: Different arrangements of NumPy arrays will be faster or slower for different algorithms. [See documentation](https://docs.dask.org/en/stable/array-chunks.html). We choose a chunk size of $(0.75\cdot\text{worker RAM, number of features})$. Because we will use repeately the features for calculating distances, but the
  - The number of threads per worker node is low due to the pure Python loop on the Lloyd's implementation, which because of Python's Global Interpreter Lock cannot be parallelized.
  - Use of DataFrames VS Array given the sklearn fetches a DataFrame, considering the time to turn it into an Array, but DataFrame doesn't implement pairwise. Array is better option in the end as sorting + filter is cheaper than groupby.
- ### TODO: **_k-means||_** optimizations considered:
    - Once $\ |x_i-c_j|^2 $ is computed for a $\ (x_i \in X,\text{ } c_j \in C) $ pair, and taken into $\ \phi_X(C) $, we don't need to compute it again. Moreover, we only need to store it if it's the minimun distance so far.
    - Original Dask implementation doesn't follow Step 7 of the **_k-means||_** and just goes to a regular **_k-means++_** with the candidate centroids in Step 8, arguing that the original paper wording remains ambiguous. [See link](https://github.com/dask/dask-ml/blob/main/dask_ml/cluster/k_means.py#L448). We understand two possibilities for these _weighting_ of the candidate centroids mentioned in the paper: either choose the initial centroid _w.p._ $\frac{w_c}{\sum_cw_c} $ or run **_k-means++_** with $\frac{w_c d^2(c, C_f)}{\sum_cw_c d^2(c, C_f)} $ as the distribution. We implement the latter.
    - It's possible that the algorithm doesn't return a set of candidate centroids $\ C $ with $\ |C| \geq k $. We have two options once this happens: either choose randomly points or run additional iterations of the algorithm until $\ |C| \geq k $ is satisfied. We chose the former.
## Benchmarking
- ### TODO: Results of performance benchmarking 
    - With increasing number of worker nodes
    - With increasing number of threads per worker node
    - With increasing size of chunks of the dask array.
- ### TODO: Comparison with Dask's ``KMeans()``
Below is the Dask **_k-means||_** API, from [dask's examples](https://examples.dask.org/machine-learning/training-on-large-datasets.html?highlight=k%20means).
``` python
km = dask_ml.cluster.KMeans(n_clusters=3, init_max_iter=2, oversampling_factor=10)
km.fit(X)
```
# Setting Dask Workers and Dask Scheduler cluster
## Scheduler 
Let the scheduler be the machine 10.67.22.164.
``` bash
$dask-scheduler
```
 ## Workers
 ``` bash
 $dask-worker  tcp://10.67.22.164:8786 —nworkers <n_workers>
```
## Exposing the Scheduler notebook to our machine
- Open the folder with the notebook in the scheduler VM
``` bash
 jupyter notebook Main.ipynb --allow-root --no-browser --port=8080
```
- Open a Terminal in your own machine 
 ``` bash
 ssh -L 8080:localhost:8080 -J your_username@gate.cloudveneto.it -L 8080:localhost:8080 root@10.67.22.164  

```
- Get the link of the notebook from the scheduler VM or open in your browser localhost:8080 and paste the notebook token
