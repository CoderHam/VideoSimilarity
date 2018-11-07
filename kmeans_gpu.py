#! /usr/bin/env python3
import numpy as np
import time
import faiss
import sys

import imageio
from skimage import transform
from collections import Counter

# Get command-line arguments
# k = int(sys.argv[1])

# n = 10**6
# d = 200
# x = np.random.random((n, d)).astype('float32')

def load_image(img_path, resize=True):
    tmp_img = imageio.imread(img_path)
    if resize:
        return transform.resize(image=tmp_img,output_shape=(200,200))
    return tmp_img

# x = x.reshape(x.shape[0], -1).astype('float32')

def train_kmeans(x, k):
    "Runs kmeans on one or several GPUs"
    d = x.shape[1]
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = 20

    clus.max_points_per_centroid = 10000000

    res = faiss.StandardGpuResources()

    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0

    index = faiss.GpuIndexFlatL2(res, d, cfg)

    # perform the training
    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)

    obj = faiss.vector_float_to_array(clus.obj)
    # print("final objective: %.4g" % obj[-1])

    return centroids.reshape(k, d)

def run(image_path,clusters=10,c_size=True):
    img_x = load_image(image_path)
    img_x = img_x.reshape((img_x.shape[0] * img_x.shape[1], 3)).astype('float32')
    centroids_ = train_kmeans(img_x, clusters)
    if c_size:
        labels = compute_cluster_assignment(centroids_,img_x)
        counts = Counter(labels)
        centroids_ = (centroids_*255).astype("uint8")
        sizes = list(zip([centroids_[k] for k in counts.keys()],counts.values()))
        return centroids_, sizes
    else:
        return (centroids_*255).astype("uint8")

def compute_cluster_assignment(centroids, x):
    assert centroids is not None, "should train before assigning"
    d = centroids.shape[1]

    res = faiss.StandardGpuResources()

    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0

    index = faiss.GpuIndexFlatL2(res, d, cfg)
    index.add(centroids)
    distances, labels = index.search(x, 1)
    return labels.ravel()

run("data/images/golden1.jpg")
