## IDEA

## Requirements 

* Python 3.6
* keras - 1.2.1
* tensorfow - 1.4.*
* A GPU

## Getting the data 

This is captioned video data from MSVR dataset available online. 

## Extracting Features 

Penultimate layer of a VGG net for every frame in the video.

## Network Architecture 

Word embeddings + Video features both go into a GRU and a caption is generated for every video.

## Running the script

* Get the videos and captions from MSVR wesbite 
* Extract the features using the extract_features.py script in the spatio_temporal folder of this repo
* Run Caption_Generator.py

## Caution

* Change the batch size to fit on your machine accordingly
* Ensure the right versions of CUDA, cudnn are installed to be able to run with the listed keras and tensorflow version

## TO DO 

* Write script to predict, so far only training has been possible 
* Start with sentence similarity to be able to cluster generated captions with repo of existed captions. 
* Figure out how to get training to start from previously calculated epoch instead of starting from scratch
 
## Some Statistics 

* Machine Used : EC2 p3.xlarge instance with 30GB memeory >>>> insufficent, as results generated exceed 30GB
* Training time : about 1 hour 

