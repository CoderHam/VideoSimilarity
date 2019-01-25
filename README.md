# VideoSimilarity

This code generates feature vectors from videos in `data/videos/` folder using pre-trained models from PyTorch. It extracts specified number of frames from each video and uses PyTorch's pretrained models (Resnet18 or Alexnet) on each video frame, and saves the feature vector into an hdf5 file.

## Downloading the UCF-101 Data

First, download the dataset from UCF into the data folder:

`cd data && wget http://crcv.ucf.edu/data/UCF101/UCF101.rar`

Then extract it with `unrar e UCF101.rar` and delete the `UCF101.rar` file.

## Performing Feature Extraction

Run the `generate_features.py` script. It has the following inputs:

```
i = input, path to the videos
n = nframes, number of frames to extract from each video
d = delete, if True, deletes the extracted frames, otherwise saves it in data/video_frames/
o = output, name of the output file that contains feature vectors
m = model, name of the model to be used in feature extraction
```
For example, running the following command from the root directory

`python cnn_features/generate_features.py -i data/ -n 10 -d False -o features -m resent`

will run feature extraction on all videos in `data` directory sampling at 10 frames per video, not deleting the
extracted frames, using Resnet18 model, and save the results to `features.pickle` file.  

## Image Feature Extraction using CNN

**Resent18 Feature Extraction Experiments**

Number of Trials: 10

Image 1 Feature Vector Generation Time: 0.013s

Image 2 Feature Vector Generation Time: 0.013s

Corpus Size=3000000, Frames per Video=100

Corpus Processing Time: 1083.333 hrs, or 45.139 days

**UCF-101 Dataset Feature Extraction**

Extracting Resnet18 features (512 dimensions) from 13,320 vidoes at 10 frames per video takes about 6-8 hours
on a local machine using CPU and no parallelization.

## k-NN Using Extracted Features

In the command line from the root directory, simply run the following:

` python cnn_features/knn_query.py`

The main function of the `knn_query.py` script can be changed to modify the query settings, including the query image and the value of k.

**k-Nearest Neighbor Performance**

k-NN query time across 134,521 frames for a single query is approximately ~0.3 seconds.
