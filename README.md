# VideoSimilarity

## My experiments on similarity metrics for videos (and images)

### Color based similarity

A Faster - GPU based implementation of k-means - can be useful for getting the dominant color

I used Facebook’s [**faiss**](https://github.com/facebookresearch/faiss) GPU implementation and compared it with scikit-learn’s vanilla [k-means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). Below is a table of speeds.

**Useful Links:**
1. [Compares Python, C++ and Cuda implementations of k-means](http://www.goldsborough.me/c++/python/cuda/2017/09/10/20-32-46-exploring_k-means_in_python,_c++_and_cuda).

![Runtime of k-means clustering on a 200x200x3 image (seconds)](https://github.com/CoderHam/VideoSimilarity/blob/master/plots/chart.png)

As we can see, using a GPU gives a massive performance boost in this case from **13x** to nearly **500x** as the cluster size (k) increases from **5** to **50**. We see sklearn takes exponentially more time and a GPU can definitely speed up this process. Although the algorithms are not exactly the same, the performance and quality of results are comparable and the speedup is worth the minor loss in accuracy.

**PS:** _In all experiments, the runtime includes the time to load the image and cluster it._

Then we can perform a similarity search using these dominant colors to find similar images and by concatenating dominant colors we can find similar videos.

**Explanation:**

We get such a **1x100** image for each of the images we cluster i.e. we cluster the resized image of size **200x200** and create a smaller image of size **100 (0.25%)** that contains its dominant colors in descending order of dominance.

![Bar of dominant colors ordered by cluster size](https://github.com/CoderHam/VideoSimilarity/blob/master/plots/clustered_bar.png)

The next step is to concatenate these **100** pixel images for each frame (samples at **n** fps from the video of length **L** seconds) to create a new image of size **L x n x 100** pixel image that will represent the entire video.

This compressed image representation is thereafter used to find similar images using either the **MSE** or **SSIM** similarity metric or by using the **KNN** algorithm.

### KNN similarity search

We can cluster smaller images/features and use the faiss - GPU implementation instead of sklearn.

**TODO: ** test with sklearn and benchmark

### Using Wavelet image hash for similarity search:

https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5 - Uses [imagehash](https://pypi.org/project/ImageHash/), a python library to compute 1 of 4 different hashes and use hashes for comparison

## Image Feature Extraction using CNN

Initial experiment results with Resent18:

Number of Trials: 10
Image 1 Feature Vector Genration Time: 0.013s
Image 2 Feature Vector Genration Time: 0.013s

Corpus Size=3000000, Frames per Video=100
Corpus Processing Time: 1083.333 hrs, or 45.139 days

Initial experiment results with Alexnet:

N/A
