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
        print('using alexnet...')
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
    return my_embedding


features = {}
# loop through all the videos in input directory
for path, subdirs, files in os.walk(input):
    for name in files:
        # get directory paths and video lengths
        video_length = int(float(FFProbe(os.path.join(path, name)).video[0].duration)) + 1
        fps = int(n_frames)/video_length
        video_path = os.path.join(path, name)
        f = FFMPEGFrames.FFMPEGFrames(output)
        f.extract_frames(os.path.join(path, name), fps)
        # get feature vectors for each frame image
        frames_path = f.full_output
        frames = os.listdir(frames_path)
        model = 'resnet'
        features[video_path] = [get_vector(os.path.join(frames_path, frame), model) for frame in frames]
        # delete rest of the files after extracting image features
        if delete:
            shutil.rmtree(f.full_output)

# save feature vectors
with open('features.pickle', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
