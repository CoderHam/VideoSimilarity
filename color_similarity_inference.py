import numpy as np
import h5py
import subprocess
import knn_cnn_features
import kmeans_gpu
import glob

def extract_vid2img_from_vid(vid_path):
    frame_list = glob.glob('data/tmp/'+vid_path.split('/')[-1].split('.')[0]+'/*.jpg')
    vid_img = None
    for fr in frame_list:
        _, bar_img = kmeans_gpu.run(fr,clusters=10)
        if type(vid_img) is not np.ndarray:
            vid_img = bar_img
        else:
            vid_img = np.concatenate((vid_img, bar_img), axis=0)
    return vid_img

def load_color_data_ucf():
    h5f = h5py.File('color_UCF_vid2img.h5', 'r')
    color_vid2imgs = np.array(h5f['color_vid2imgs']).astype('float32')
    color_labels = np.array([fl.decode() for fl in h5f['vid_labels']])
    h5f.close()
    return color_vid2imgs, color_labels

def get_ordered_unique(listed):
    seen = set()
    seen_add = seen.add
    ordered_listed = [x for x in listed if not (x in seen or seen_add(x))]
    return ordered_listed

def similar_color_ucf_video(vid_path, k=10, dist=False, verbose=False):
    vid_feature_vector = extract_vid2img_from_vid(vid_path)
    # color_vid2imgs, color_labels = load_color_data_ucf()
    vid_feature_vector = vid_feature_vector.flatten()[np.newaxis,].astype('float32')
    distances, feature_indices = knn_cnn_features.run_knn_features(\
        color_vid2imgs, test_vectors=vid_feature_vector,k=k, dist=True)
    if verbose:
        print(color_labels[feature_indices][0])
    if dist:
        return list(distances[0]), list(color_labels[feature_indices][0])
    else:
        return list(color_labels[feature_indices][0])

color_vid2imgs, color_labels = load_color_data_ucf()

# test
# import time
# start = time.time()
# for i in range(10):
#     similar_color_ucf_video('data/UCF101/v_ApplyEyeMakeup_g01_c01.avi', verbose=True)
# print((time.time()-start)/10)
# 2.293074941635132 seconds
