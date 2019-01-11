## Getting the data

First, download the dataset from UCF into the `data` folder:

`cd data && wget http://crcv.ucf.edu/data/UCF101/UCF101.rar`

Then extract it with `unrar e UCF101.rar`.

Next, create folders (still in the data folder) with `mkdir train && mkdir test && mkdir sequences && mkdir checkpoints`.

Now you can run the scripts in the data folder to move the videos to the appropriate place, extract their frames and make the CSV file the rest of the code references. You need to run these in order. Example:

`python create_train_test.py`

`python extract_frames.py`

## Extracting features

To be able to the Multi Layer Perceptron and LSTM mdoels, we need to extract features from the images with the CNN. This is done by running `extract_features.py`. 

For all 101 features, this script took around 2 hours on a p2.xlarge EC2 instance on AWS which had a GPU and about 16GB of space(which spilled over and had to be rescaled) 

WARNING : The extracted train+test folders will be around 21GB 

There is an option to extract only a certain number of features as well, which can be set with `class_limit` variable within the script.

## Training Models 

Run `train.py` to run either : Long term Recurrent Convolutional Network, LSTM or a Multi Layer Perceptron network.

SOME NOTES: 
- The number of classes in the `train.py` script has been hardcoded to 10 which can be set to the number as desired from 1-101. 
- The dimensions for the image are hardcoded as per what has been extracted earlier using `extract_frames.py` for now since the UCF101 is a standard dataset and must be changed for other datasets accordingly.
- The hyperparameters such as : dropout rate, number of strides, number of layers, activation functions etc have been set to the best possible values post hyper parameter tuning 
- Each model takes a few hours to run even on a GPU given the suze of the dataset.
