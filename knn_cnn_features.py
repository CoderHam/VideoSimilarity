import numpy as np
import faiss
import sys

import knn_gpu

def run_knn_features(feature_vectors, test_vectors=None, k=5, flat=True,
                     verbose=False, dist=False, gpu=False):
    nb, d = feature_vectors.shape
    if type(test_vectors) is not np.ndarray:
        if flat:
            D, I = knn_gpu.knn_flat(feature_vectors, feature_vectors, d, k, verbose, gpu)
        else:
            D, I = knn_gpu.knn_ivf(feature_vectors, feature_vectors, d, k, verbose, gpu)
    else:
        if flat:
            D, I = knn_gpu.knn_flat(feature_vectors, test_vectors, d, k, verbose, gpu)
        else:
            D, I = knn_gpu.knn_ivf(feature_vectors, test_vectors, d, k, verbose, gpu)

    if dist:
        return D, I
    else:
        return I
