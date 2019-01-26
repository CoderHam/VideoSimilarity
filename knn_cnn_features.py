import numpy as np
import time
import faiss
import sys

import imageio
from skimage import transform
from collections import Counter
import cv2
import glob
import knn_gpu

def run_knn_features(feature_vectors,test_vectors=None,k=5,flat=True,verbose=False):
    nb, d = feature_vectors.shape
    if type(test_vectors) is not np.ndarray:
        if flat:
            D, I = knn_gpu.knn_flat(feature_vectors, feature_vectors, d, k, verbose)
        else:
            D, I = knn_gpu.knn_ivf(feature_vectors, feature_vectors, d, k, verbose)
    else:
        if flat:
            D, I = knn_gpu.knn_flat(feature_vectors, test_vectors, d, k, verbose)
        else:
            D, I = knn_gpu.knn_ivf(feature_vectors, test_vectors, d, k, verbose)

    return I
