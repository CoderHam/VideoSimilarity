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

def run_knn_features(feature_vectors,k,flat=True):
    nb, d = feature_vectors.shape
    print("Number of records:",nb, "\nNumber of dimensions:",d)
    if flat:
        D, I = knn_gpu.knn_flat(feature_vectors, feature_vectors, d, k)
    else:
        D, I = knn_gpu.knn_ivf(feature_vectors, feature_vectors, d, k)

    return I
