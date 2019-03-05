"""
This file performs similarity inference using 3D CNN.
"""
import os
import sys
import json
import pickle
import h5py
import subprocess
import numpy as np
import torch
from torch import nn
import time
import h5py

sys.path.insert(0, './3d-cnn')
from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video

import knn_cnn_features

def extract_features():
    """
    This function extracts 3D CNN features for all the videos in the dataset
    using ResNet-34 based model.

    The extraction process takes ~8 hours on AWS GPU instance, and requires
    around 32GB of memory.
    """
    t_start = time.time()
    os.chdir('3d-cnn/')
    os.system('python main.py --input input --video_root videos --output output.json --model resnet-34-kinetics.pth --mode feature --sample_duration 8')
    t_end = time.time()
    print('Total 3D CNN feature extraction time: {}s'.format(round(t_end - t_start, 2)))


def extract_features_from_vid(video_path):
    """
    This function extracts 3D CNN features for a single query video.
    """
    opt = parse_opts()
    opt.input = video_path
    opt.model = './3d-cnn/resnet-34-kinetics.pth'
    opt.mode = 'feature'
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 8
    opt.n_classes = 400

    model = generate_model(opt)
    # print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    outputs = []
    class_names = []
    with open('./3d-cnn/class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    if os.path.exists(opt.input):
        subprocess.call('mkdir tmp', shell=True)
        subprocess.call('ffmpeg -loglevel panic -i {} tmp/image_%05d.jpg'.format(video_path),shell=True)
        result = classify_video('tmp', video_path, class_names, model, opt)
        outputs.append(result)
        subprocess.call('rm -rf tmp', shell=True)
    else:
        print('{} does not exist'.format(video_path))

    return np.array(outputs[0]['clips'][0]['features']).astype('float32')

def pickle_to_hdf5(pickle_file='./cnn3d_features/ucf101_3dcnn_features'):
    features = pickle.load(open(pickle_file, 'rb'))
    feature_vectors = []
    video_labels = []
    for video in features:
        feature_vectors.append(video['clips'][0]['features'])
        video_labels.append(video['video'])
    h5f = h5py.File('cnn3d_UCF_features.h5', 'w')
    h5f['feature_vectors'] = np.array(feature_vectors).astype('float32')
    h5f['vid_labels'] = np.array(video_labels, dtype='S')
    h5f.close()

def process_output(output_filename):
    """
    This function processes the extracted 3D CNN features for dataset to be used
    in kNN search algorithm.
    """
    # load extracted features json file
    try:
        with open(output_filename, 'r') as f:
            features = json.load(f)
    except:
        file = open(output_filename, 'rb')
        features = pickle.load(file)
    # initialize feature vector list and mapping list
    feature_vectors = []
    video_labels = []
    # populate the above
    for i, video in enumerate(features):
        feature_vectors.append(video['clips'][0]['features'])
        video_labels.append(video['video'])
    # convert to numpy arrays and float32
    feature_vectors = np.array(feature_vectors)
    feature_vectors = feature_vectors.astype('float32')
    return feature_vectors, video_labels


def save_output(feature_vectors, video_labels, output_filename):
    """
    This function saves the processsed feature vectors and video labels into
    h5py file.
    """
    # define paths and filenames
    output_path = './cnn3d_features/'
    feature_vectors_filename = 'feature_vectors_' + output_filename
    video_labels_filename = 'video_labels_' + output_filename
    # save feature vectors and video labels
    if not os.path.exists(os.path.join(output_path, feature_vectors_filename)):
        with h5py.File(os.path.join(output_path, feature_vectors_filename), 'w') as fv:
            fv.create_dataset("ucf101_cnn3d_feature_vectors",  data=feature_vectors)
    else:
        print(feature_vectors_filename + ' already exists!')
    if not os.path.exists(os.path.join(output_path, video_labels_filename)):
        video_labels = [label.encode("ascii", "ignore") for label in video_labels]
        with h5py.File(os.path.join(output_path, video_labels_filename), 'w') as vl:
            vl.create_dataset("ucf101_cdd3d_video_labels",  data=video_labels)
    else:
        print(video_labels_filename + ' already exists!')


def load_feature_vectors(feature_vectors_filename, features_path):
    """
    This function loads the processed 3D CNN feature vectors for kNN querying.
    """
    if os.path.exists(os.path.join(features_path, feature_vectors_filename)):
        with h5py.File(os.path.join(features_path, feature_vectors_filename), 'r') as fv:
            print('loading feature vectors...')
            feature_vectors = fv['ucf101_cnn3d_feature_vectors'][:]
        return feature_vectors
    else:
        print(feature_vectors_filename + ' does not exist!')


def load_video_labels(video_labels_filename, features_path):
    """
    This function loads the processed video labels.
    """
    if os.path.exists(os.path.join(features_path, video_labels_filename)):
        with h5py.File(os.path.join(features_path, video_labels_filename), 'r') as vl:
            print('loading video labels...')
            video_labels = vl['ucf101_cdd3d_video_labels'][:]
        return video_labels
    else:
        print(video_labels_filename+' does not exist!')


def similar_cnn3d_ucf_video(video_path, k=10, dist=False, verbose=False, gpu=True, query_features=None, newVid=False):
    """
    This function extracts features from the query video and performs kNN similarity search.
    """
    try:
        if newVid:
            if query_features is not np.ndarray:
                query_features = extract_features_from_vid(video_path)
            assert query_features.shape == (512,)
            distances, feature_indices = knn_cnn_features.run_knn_features(feature_vectors,
                test_vectors=query_features[np.newaxis,:], k=k, dist=True, flat=True, gpu=gpu)
        else:
            query_features = feature_vectors[np.where(video_labels == np.str_(video_path.split('/')[-1].split(".")[-2]+".avi"))[0]]
            distances, feature_indices = knn_cnn_features.run_knn_features(feature_vectors,
                test_vectors=query_features, k=k, dist=True, gpu=gpu)
        del query_features
        if verbose:
            print(video_labels[feature_indices][0])
        if dist:
            return list(distances[0]), list(map(str,video_labels[feature_indices[0]]))
        else:
            return list(feature_indices[0])
    except:
        print('No video found!')


def output_query_results(knn_indicies, video_labels, query_video_path):
    """
    This function outputs results of a kNN query.
    """
    # print('\nQuery Completed in {}s'.format(query_time))
    print('\nQuery Video:\n', query_video_path)
    print('\n')
    print('Top Similar Videos:')
    for i, ind in enumerate(knn_indicies):
        print(i+1, ': ', video_labels[ind])


def full_build():
    """
    This function runs a full data processing build from raw extracted features.
    """
    # s_time = time.time()
    # print('Starting full 3D CNN feature extraction...')

    # feature extraction (run this only if full feature extraction needed)
    # print('Extracting features from UCF101 dataset...')
    # feature_vectors = extract_features()

    # data processing
    s_processing = time.time()
    print('Processing raw extracted features...')
    feature_vectors, video_labels = process_output('./cnn3d_features/ucf101_3dcnn_features')
    e_processing = time.time()
    print('Raw features processing completed in {}s.'.format(round(e_processing - s_processing, 2)))

    # save and load processed features
    print('Saving processed features...')
    save_output(feature_vectors, video_labels, 'cnn3d_ucf101')
    print('Loading processed features...')
    s_loading = time.time()
    feature_vectors = load_feature_vectors('feature_vectors_cnn3d_ucf101',
                                            './cnn3d_features/')
    video_labels = load_video_labels('video_labels_cnn3d_ucf101',
                                        './cnn3d_features/')
    e_loading = time.time()
    print('Loading processed features completed in {}s.'.format(round(e_loading - s_loading, 2)))

    # test_query
    query_video = video_labels[0]
    query_features = feature_vectors[0]
    query_features = np.reshape(query_features.astype(np.float32), (1, -1))
    cnn3d_dist, cnn3d_indices = similar_cnn3d_ucf_video(query_video,
            k=3, dist=True, verbose=False, gpu=True, query_features=query_features)
    output_query_results(cnn3d_indices, video_labels, video_labels[0])
    print('\nFull build successfully completed!')


def quick_query_test():
    """
    Test Script (CPU):

    Tests full data pipeline using sample of UCF 101 dataset. It does not
    perform feature extraction for a single video, instead using one of the
    extracted features as query video features.
    """
    if not use_h5:
        # data processing
        feature_vectors, video_labels = process_output('./cnn3d_features/ucf101_3dcnn_features_sample.json')

        # save and load processed features
        save_output(feature_vectors, video_labels, 'cnn3d_ucf101')
        feature_vectors = load_feature_vectors('feature_vectors_cnn3d_ucf101',
                                                './cnn3d_features/')
        video_labels = load_video_labels('video_labels_cnn3d_ucf101',
                                            './cnn3d_features/')
    # test_query
    # query_video_path = 'v_ApplyEyeMakeup_g04_c03.avi'
    query_video = video_labels[0]
    query_features = feature_vectors[0]
    query_features = np.reshape(query_features.astype(np.float32), (1, -1))
    cnn3d_dist, cnn3d_indices = similar_cnn3d_ucf_video(query_video,
        k=3, dist=True, verbose=False, gpu=True, flat=True, query_features=query_features)
    output_query_results(cnn3d_indices, video_labels, video_labels[0])
    print('\nTest successfully completed!')

def load_cnn3d_feature_data_ucf():
    h5f = h5py.File('cnn3d_UCF_features.h5', 'r')
    feature_vectors = np.array(h5f['feature_vectors']).astype('float32')
    video_labels = np.array([fl.decode() for fl in h5f['vid_labels']])
    h5f.close()
    return feature_vectors, video_labels

use_h5=True
if use_h5:
    feature_vectors, video_labels = load_cnn3d_feature_data_ucf()

# import time
# start = time.time()
# for i in range(5):
#     similar_cnn3d_ucf_video('data/UCF101/v_ApplyEyeMakeup_g01_c01.webm', verbose=True, newVid=True)
# print((time.time()-start)/5)
# 2.168000817298889 seconds (1 s if not new video)

# test or full build
# quick_query_test()
# full_build()
