"""
This script extracts features from videos and saves it to a Python pickle file.

i = input, path to the videos
n = nframes, number of frames to extract from each video
d = delete, if True, deletes the extracted frames
o = output, name of the output hd5f file
m = model, name of the model to be used in feature extraction
"""

# import torch and other libraries
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from IPython.display import display # to display images

# other imports
import time
import numpy as np
import shutil
import os
import argparse
import pickle
from ffprobe3 import FFProbe

# custom imports
import FFMPEGFrames

# input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-n", "--nframes", required=True)
ap.add_argument("-d", "--delete", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

input = args["input"]
n_frames = args["nframes"]
delete = args["delete"]
output = args["output"]
model = args["model"]

# Load the pretrained model
resnet_model = models.resnet18(pretrained=True)
alexnet_model = models.alexnet(pretrained=True)
# Use the model object to select the desired layer
resnet_layer = resnet_model._modules.get('avgpool')
alexnet_layer = alexnet_model._modules.get('classifier')

# Set model to evaluation mode
resnet_model.eval()
alexnet_model.eval()

# image scaler to 224 x 224 pixels
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

# function to generate feature vector for a single image
def get_vector(image_name, model):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    if model == 'resnet':
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(512)
        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        # 5. Attach that function to our selected layer
        h = resnet_layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        resnet_model(t_img)
        h.remove()
    elif model == 'alexnet':
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'classifier' layer has an output size of 1000
        my_embedding = torch.zeros(1000)
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        # 5. Attach that function to our selected layer
        h = alexnet_layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        alexnet_model(t_img)
        h.remove()
    # 8. Return the feature vector
    return np.array(my_embedding)

# extract frames and extract features for each frame
features = []
ind2path_labels = {}
path2ind_labels = {}
ind = 0
# loop through all the videos in input directory
for path, subdirs, files in os.walk(input):
    for name in files:
        # print(os.path.join(path, name))
        # get directory paths and video lengths
        video_length = int(float(FFProbe(os.path.join(path, name)).video[0].duration)) + 1
        fps = int(n_frames)/video_length
        video_path = os.path.join(path, name)
        frames_output = 'data/video_frames'
        # frames_output = '/mnt/e/ucf_101_frames/'
        f = FFMPEGFrames.FFMPEGFrames(frames_output)
        f.extract_frames(os.path.join(path, name), fps)
        # get feature vectors for each frame image
        frames_path = f.full_output
        frames = os.listdir(frames_path)
        feature_matrix = np.array([get_vector(os.path.join(frames_path, frame), model) for frame in frames])
        for frame in frames:
            feature_matrix = np.array(get_vector(os.path.join(frames_path, frame), model))
            full_path = os.path.join(video_path, frame)
            features.append(feature_matrix)
            ind2path_labels[ind] = full_path
            path2ind_labels[full_path] = ind
            ind += 1
        # delete rest of the files after extracting image features
        if delete == True:
            print('deleting frames subdirectory...')
            shutil.rmtree(f.full_output)

# convert to numpy matrix and save
features_file = [np.array(features), ind2path_labels, path2ind_labels]
output_filename = output + '.pickle'
with open(output_filename, 'wb') as handle:
    pickle.dump(features_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
