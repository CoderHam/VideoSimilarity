# VideoSimilarity

This code generates feature vectors from videos in `data/videos/` folder using pre-trained models from PyTorch. It extracts specified number of frames from each video and uses PyTorch's pretrained models (Resnet18 or Alexnet) on each video frame, and saves the feature vector into an hdf5 file.

## How to perform feature extraction

`python generate_features.py -i data/videos/ -n n_frames -d True -o features -m resent`

## Image Feature Extraction using CNN

**Initial experiment results with Resent18:**

Number of Trials: 10

Image 1 Feature Vector Generation Time: 0.013s

Image 2 Feature Vector Generation Time: 0.013s

Corpus Size=3000000, Frames per Video=100

Corpus Processing Time: 1083.333 hrs, or 45.139 days

**Initial experiment results with Alexnet:**

N/A
