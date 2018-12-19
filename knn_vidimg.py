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
import re
from sklearn.metrics import accuracy_score

def load_image(img_path, resize=True):
    tmp_img = imageio.imread(img_path)
    if resize:
        return transform.resize(image=tmp_img,output_shape=(200,200),anti_aliasing=True, mode='constant')
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

    return I

def get_class_from_path(vid_path):
    return re.search('vid2img/(.*)_', vid_path, re.IGNORECASE).group(1)

def get_classes_from_indices(vid2img_list,vidIndices,label=True):
    classes = []
    for i,vidIndex in enumerate(vidIndices):
        # classes = [get_class_from_path(vid2img_list[j]) for j in vidIndex]
        voted_classes = [j//10 for j in vidIndex]
        if label:
            classes.append(Counter(voted_classes).most_common(1)[0][0])
        else:
            classes.append(voted_classes)
    return classes

def get_accuracy(vid2img_list,vidIndices):
    pred = get_classes_from_indices(vid2img_list,vidIndices)
    actual = [j//10 for j in range(70)]
    return accuracy_score(actual, pred)

# vid2img_list = sorted(glob.glob('data/vid2img/*.jpg'))
# vidIndices = run_vid2img(vid2img_list,3)
# print("Accuracy:",get_accuracy(vid2img_list,vidIndices))
