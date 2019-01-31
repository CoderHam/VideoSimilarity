# VideoSimilarity

## Experiments on similarity metrics for videos (and images)

### 1. Color based similarity - [color_similarity.ipynb](https://github.com/CoderHam/VideoSimilarity/blob/master/color_similarity.ipynb)

A Faster - *GPU* based implementation of **k-means clustering** - is used for getting the dominant color.

I used Facebook’s [**faiss**](https://github.com/facebookresearch/faiss) GPU implementation and compared it with scikit-learn’s vanilla [k-means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). Below is a table of speeds.

**Useful Links:**
1. [Compares Python, C++ and Cuda implementations of k-means](http://www.goldsborough.me/c++/python/cuda/2017/09/10/20-32-46-exploring_k-means_in_python,_c++_and_cuda).

![Runtime of k-means clustering on a 200x200x3 image (seconds)](https://github.com/CoderHam/VideoSimilarity/blob/master/plots/chart.png)

|Number of clusters (k)|Runtime with CPU (s)|Runtime with GPU (s)|
|---|---|--|
|5|4|0.3|
|10|18|0.3|
|20|45|0.3|
|50|141|0.3|

As we can see, using a GPU gives a massive performance boost in this case from **13x** to nearly **500x** as the cluster size (k) increases from **5** to **50**. We see sklearn takes exponentially more time and a GPU definitely speeds up this process. Although the algorithms are not exactly the same, the performance and quality of results are comparable and the speedup is worth the minor loss in accuracy.

**PS:** _In all experiments, the runtime includes the time to load the image and cluster it._

Then we perform a similarity search using these dominant colors to find similar colored images and by concatenating dominant colors for multiple frames, we find similar colored videos.

**Explanation:**

We get such a **1x100** image for each of the images we cluster i.e. we cluster the resized image of size **200x200** and create a smaller image of size **100 (0.25%)** that contains its dominant colors in descending order of dominance.

![Bar of dominant colors ordered by cluster size](https://github.com/CoderHam/VideoSimilarity/blob/master/plots/clustered_bar.png)

The next step is to concatenate these **100** pixel images for each frame (samples at **n** fps from the video of length **L** seconds) to create a new image of size **L x n x 100** pixel image that will represent the entire video.

This compressed image representation is thereafter used to find similar images using either the **MSE** or **SSIM** similarity metric or by using the **KNN** algorithm.

PS: For the current experiments, **L** and **n** is variable but I have used a script to extract a fixed number of frames **(20)** for each video. Thus the new image representation is **20 x 100 x 3** [since there are **3** channels (RGB)].

I have extracted this image representation for the videos in [`data/videos`](https://github.com/CoderHam/VideoSimilarity/tree/master/data/videos) and will now use KNN to return k-similar neighbors.

### KNN similarity search on (vid2img representations)

We cluster smaller images / features and use the faiss - *GPU* implementation instead of sklearn and store them in [`data/vid2img`](https://github.com/CoderHam/VideoSimilarity/tree/master/data/vid2img).

For **10 x 7 = 70** videos, the process took **437 s** to execute i.e. approx **6.2 s** for video. This process includes:

1. Extracting frames from videos and writing them to disk.
2. Clustering the extracted frames (20 per video).
3. Converting the histogram of clusters into an image and writing to disk.

Thereafter, I ran the KNN GPU implementation on the image representations in [`data/vid2img`](https://github.com/CoderHam/VideoSimilarity/tree/master/data/vid2img).

The dimensionality is **6000** since (*20 x 100 = 2000* pixels and RBG (*3*) values for each pixel).

The runtime for the KNN search with **K=3** (build and run on all videos) for this subset of **70** videos is **132 ms** and we can be sure that this will scale effectively based on previous experiments.

### 2. Feature based similarity - [feature_similarity.ipynb](https://github.com/CoderHam/VideoSimilarity/blob/master/feature_similarity.ipynb)

The [extract_features.py](extract_features.py) script (Pytorch), extracts CNN features/embeddings using a pre-trained Resnet50 model. The feature vector is **2048** dimensional. Since the UCF101 dataset has a median video length of **8** seconds.

For **1000** image feature vectors takes the KNN search with K=3 takes **94.9** ms. We extract the features for all **13320** videos with **8** uniformly sampled frames each and extract feature vectors from each such frame. The process takes nearly 1 hour and the results are stored in [features_UCF_resnet50.h5](https://drive.google.com/open?id=1h6Jv28NXkD-_Hyb3XXjomDtu3jLKKdb6). This has the **8 x 2048** matrix for each video with the video path as the key (data/UCF101/<video_name>.avi).

Since we need the merged numpy array we also create a merged feature vector of shape **106557 x 2048** and store it in [merged_features_UCF_resnet50.h5](https://drive.google.com/open?id=1bWBQ98mlcbt_sr4ipAlM0mdWy7PkdSgZ). The features are stores in a `feature_vectors` and corresponding labels are stored in `feature_labels`.

Performing the KNN similarity search with **K=3** takes **18** seconds. (This includes **100,000** queries as well, one for each frame).

The **class-wise accuracy** is then calculated and it comes to **96.4%**.

**KNN feature similarity tests:-**

**1. Validation using train-test split (70:30):**

Therefore we see that by taking a **70:30** split for train and test we still get a class-wise accuracy of **91.6%** (as compared to **96.4%** for all the data) for the **101** classes. Which is still good!

Moreover the time reduces from **17.3** seconds to **3.9** seconds when queries reduces from **100,000** to **30,000**.

**2. Number of queries = 100, 10, 1**

The time reduces to approx **0.2** seconds for 100, 10 and 1 query. This will allow us to run our KNN similarity search in **real-time**!

|No. of Queries|Runtime (s)|
|---|---|
|1|0.2|
|10|0.2|
|100|0.2|

**3. With K = 10, 100, 200**

|Value of K|Runtime (s)|
|---|---|
|3|0.2|
|10|0.2|
|100|0.2|
|200|0.2|

As we can see the KNN similarity search scales well with the value of **K**. A single query on the **100,000** datapoints of dimensionality **2048** takes approximately **0.2** s each for **K = 3, 10, 100, 200**. Still runnable in real-time!

### 3. Sound based similarity - [sound_similarity.ipynb](https://github.com/CoderHam/VideoSimilarity/blob/master/sound_similarity.ipynb)

The word is based on the `audioset` dataset and `VGGish` model trained by Google (Tensorflow). The pipeline follow subsampling of audio to a standard form followed by creating a `log mel-spectrogram` of size **(96, 64)** for each second. This is then fed into the pre-trained VGGish model that returns a **128** dimensional embedding for each second of audio. It is important to not that all audio clips in this dataset are on **10** seconds each. We use the balanced train split of the audioset data to test the performance which comprises of **21,782** audio clips.

The embeddings from the `.tfrecord` files are read from disk and preprocessed. Since we need the merged numpy array we created a merged audio embedding matrix of shape **21,782 x 10 x 128** and stored it in [audioset_balanced_features_vggish.h5](https://drive.google.com/file/d/1oepvdCfpw8RuAk8AppIHvKUxqTBz5S1N/view?usp=sharing). The features are stores in a `audio_embeddings` and corresponding label(s) for each audio clip is stored in [audioset_balanced_labels.npy](https://drive.google.com/file/d/1xsL8IQAZ9i8-1AIYCd5FP0Vb9Meixbro/view?usp=sharing).

Performing the KNN similarity search with **K=3** takes **0.5** seconds. (This includes **20,000** queries as well, one for each audio clip).

The **class-wise accuracy** is then calculated and it comes to **47.3%** (for **448** classes when assuming only first label).

### Extra - Using Wavelet image hash for similarity search (Not currently using):

https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5 - Uses [imagehash](https://pypi.org/project/ImageHash/), a python library to compute 1 of 4 different hashes and use hashes for comparison
