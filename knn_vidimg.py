#! /usr/bin/env python3
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

def load_image(img_path, resize=True):
    tmp_img = imageio.imread(img_path)
    if resize:
        return transform.resize(image=tmp_img,output_shape=(200,200))
    return tmp_img

def build_vector(vid2img_list):
    img_vectors = None
    for vi in vid2img_list:
        img_x = load_image(vi,resize=False)
        img_x = img_x.flatten().astype('float32')
        if img_vectors is None:
            img_vectors = img_x
        else:
            img_vectors = np.vstack((img_vectors, img_x))
    return img_vectors

def run_vid2img(vid2img_list,k,flat=True):
    img_vectors = build_vector(vid2img_list)
    nb, d = img_vectors.shape
    print("Number of records:",nb, "\nNumber of dimensions:",d)
    if flat:
        D, I = knn_gpu.knn_flat(img_vectors, img_vectors, d, k)
    else:
        D, I = knn_gpu.knn_ivf(img_vectors, img_vectors, d, k)

    print(vid2img_list[0],":",[vid2img_list[i] for i in I[0]])
    return I

# vid2img_list = sorted(glob.glob('data/vid2img/*.jpg'))
# run_vid2img(vid2img_list,3)
