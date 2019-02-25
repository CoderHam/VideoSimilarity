import numpy as np
import faiss


## Using a flat index
def knn_flat(vector_x, queries_x, d, k, verbose, gpu=True):
    index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index
    if gpu:
        res = faiss.StandardGpuResources()  # use a single GPU
        index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index_flat.add(vector_x)

    if verbose:
        print("Running with Flat index for", index_flat.ntotal,
             "records of with dimensionality", d)
    # we want to see the k nearest neighbors
    D, I = index_flat.search(queries_x, k) # actual search
    return D, I


# Using an IVF index
def knn_ivf(vector_x, queries_x, d, k, verbose, gpu=True):
    nlist = 100
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # here we specify METRIC_L2, by default it performs inner-product search
    # make it an IVF GPU index
    if gpu:
        res = faiss.StandardGpuResources()  # use a single GPU
        index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)

    assert not index_ivf.is_trained
    index_ivf.train(vector_x)        # add vectors to the index
    assert index_ivf.is_trained
    index_ivf.add(vector_x)          # add vectors to the index

    if verbose:
        print("Running with IVF index for", gpu_index_ivf.ntotal,
              "records of with dimensionality", d)
    D, I = index_ivf.search(queries_x, k)  # actual search
    return D, I


def run(k, flat=False, verbose=False):
    # create fake data
    d = 200                           # dimensionality
    nb = 1000000                      # number of records
    nq = 10                       # number of queries
    np.random.seed(1234)
    vector_x = np.random.random((nb, d)).astype('float32')
    vector_x[:, 0] += np.arange(nb) / 1000.
    queries_x = np.random.random((nq, d)).astype('float32')
    queries_x[:, 0] += np.arange(nq) / 1000.

    # perform knn search
    if flat:
        D, I = knn_flat(vector_x, queries_x, d, k,
                        verbose=verbose, gpu=True)
    else:
        D, I = knn_ivf(vector_x, queries_x, d, k,
                        verbose=verbose, gpu=True)
    # print(I[:5])                   # neighbors of the 5 first queries
    # print(list(zip(D[:5],I[:5])))

# run(10)
