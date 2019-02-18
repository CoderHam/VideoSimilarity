"""
This file performs similarity inference using 3D CNN.
"""
import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
import time

sys.path.insert(0, './3d-cnn')
from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video

def extract_features():
    """
    This function extracts 3D CNN features for all the videos in the dataset.
    """
    t_start = time.time()
    os.chdir('3d-cnn/')
    os.system('python main.py --input input --video_root videos --output output.json --model resnet-34-kinetics.pth --mode feature --sample_duration 8')
    t_end = time.time()
    print('Total 3D CNN feature extraction time: {}s'.format(round(t_end - t_start, 2)))

def extract_features_from_vid(video_path, video_filename):
    """
    This function extracts 3D CNN features for a single query video.
    """
    opt = parse_opts()
    opt.input = video_filename
    opt.video_root = video_path
    opt.model = './3d-cnn/resnet-34-kinetics.pth'
    opt.mode = 'feature'
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 8
    opt.n_classes = 400

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
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

    video_path = os.path.join(opt.video_root, video_filename)
    print(video_path)
    if os.path.exists(video_path):
        subprocess.call('mkdir tmp', shell=True)
        subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video_path),
                        shell=True)
        result = classify_video('tmp', video_filename, class_names, model, opt)
        outputs.append(result)
        subprocess.call('rm -rf tmp', shell=True)
    else:
        print('{} does not exist'.format(video_path))
    return outputs[0]


def process_output(output_filename):
    """
    This function processes the extracted 3D CNN features for dataset to be used
    in kNN search algorithm.
    """
    # load output json file
    with open(output_filename, 'r') as f:
        features = json.load(f)
    # initialize feature vector list and mapping dicts
    feature_vectors = []
    ind2video_mapping = {}
    video2ind_mapping = {}
    # populate the above
    for i, video in enumerate(features):
        feature_vectors.append(video['clips'][0]['features'])
        ind2video_mapping[i] = video['video']
        video2ind_mapping[video['video']] = i
    # convert to numpy arrays and float32
    feature_vectors = np.array(feature_vectors)
    feature_vectors = feature_vectors.astype('float32')
    return feature_vectors, ind2video_mapping, video2ind_mapping

# test script
# extract_features()
q_results = extract_features_from_vid('./3d-cnn/videos/', 'v_ApplyEyeMakeup_g04_c02.avi')
print(len(q_results['clips'][0]['features'])) # should be 512
