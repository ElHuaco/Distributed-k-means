# Distributed-k-means
The goal of this project is to implement efficiently the **_k-means||_** algorithm in the Dask distributed computing framework, and benchmark the result with some real-world standard datasets made available by sci-kit learn, _v.g._, [RCV1](https://scikit-learn.org/stable/datasets/real_world.html#rcv1-dataset) or [kddcup99](https://scikit-learn.org/stable/datasets/real_world.html#kddcup-99-dataset).
## Implementation
- ### Dask best practices considered:
  - We keep the centroid set in the scheduler node. This avoids shuffling but cannot avoid the several ``compute()`` fetch and calls to the worker nodes to calculate distances.
  - We ``persist()`` the data points stored in each worker node, as the updating will be made on the centroid set, not on the dataset. This way we avoid multiple read operations.
  - Partitions: different arrangements of NumPy array chunks will be faster or slower for different algorithms. [See documentation](https://docs.dask.org/en/stable/array-chunks.html). In our case, we choose a chunk size of $\sim(\text{features, points s.t. 0.75 RAM is used}) $, because we will use repeately the features for computing distances.
## **_k-means||_** optimizations:
- The original Dask implementation doesn't follow Step 7 of the **_k-means||_** and just goes to a regular **_k-means++_** with the candidate centroids in Step 8, arguing that the original paper wording remains ambiguous. [See link](https://github.com/dask/dask-ml/blob/main/dask_ml/cluster/k_means.py#L448). We understand two possibilities for these _weighting_ of the candidate centroids mentioned in the paper: either choose the initial centroid _w.p._  $\frac{w_c}{\sum_cw_c}$ or run **_k-means++_** with $\frac{w_c d^2(c, C_f)}{\sum_cw_c d^2(c, C_f)}$ as the distribution. We implement the latter, as it gave better results in terms of accuracy.
- It's possible that the algorithm doesn't return a set of candidate centroids $C \text{ with } |C| \geq k $. We have two options once this happens: either choose randomly points or run additional iterations of the algorithm until $|C| \geq k$ is satisfied. We chose the former, as it's the quicker alternative.
- Once $|x_i-c_j|^2$ is computed for a $(x_i \in X,\text{ } c_j \in C)$ pair, and taken into $\phi_X(C) $, we don't need to compute it again. Moreover, we only need to store it if it's the minimun distance so far. This approach, while sound in principle, fails to improve the performance in it's current implementation. Perhaps the overhead in the memory/computation trade-off inhibits the expected speed-up for the dataset sizes used.

## Benchmarking
- ### Results of performance benchmarking
    - We find an optimal value of the chunk size in the worker nodes that must optimize the memory usage with the amount of computations done per chunk.
    - With increasing number of threads per worker node we don't see significant speed-up. Perhaps the dimensionality of the benchmark dataset was not high enough to show the parallelization speed-up.

# Setting a Dask cluster with CLI
## Scheduler 
Let the scheduler be the machine 10.67.22.164.
``` bash
dask-scheduler
```
 ## Workers
 ``` bash
dask-worker  tcp://10.67.22.164:8786 —nworkers <n_workers>
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
