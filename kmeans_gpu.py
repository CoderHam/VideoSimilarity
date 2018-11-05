# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python3
import numpy as np
import time
import faiss
import sys

import imageio
from skimage import transform

# Get command-line arguments
# k = int(sys.argv[1])

# n = 10**6
# d = 200
# x = np.random.random((n, d)).astype('float32')

def load_image(img_path, resize=True):
    tmp_img = imageio.imread(img_path)
    if resize:
        return transform.resize(image=tmp_img,output_shape=(400,400))
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

def run(image_path,clusters=10):
    img_x = load_image(image_path)
    img_x = img_x.reshape((img_x.shape[0] * img_x.shape[1], 3)).astype('float32')
    centroids_ = train_kmeans(img_x, clusters)
    return (255*centroids_).astype("uint8")

# t0 = time.time()
# t1 = time.time()
#
# print("k-Means runtime: %.3f s" % (t1 - t0))
