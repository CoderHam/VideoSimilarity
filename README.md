# VideoSimilarity

## My experiments on similarity metrics for videos (and images)

### Color based similarity

A Faster - GPU based implementation of k-means - can be useful for getting the dominant color

I used Facebook’s [**faiss**](https://github.com/facebookresearch/faiss) GPU implementation and compared it with scikit-learn’s vanilla [k-means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). Below is a table of speeds.

**Useful Links:**
1. [Compares Python, C++ and Cuda implementations of k-means](http://www.goldsborough.me/c++/python/cuda/2017/09/10/20-32-46-exploring_k-means_in_python,_c++_and_cuda).

![Runtime of k-means clustering on a 800x500x3 image (seconds)](https://github.com/CoderHam/VideoSimilarity/blob/master/chart.png)

As we can see, using a GPU gives a massive performance boost in this case from **13x** to nearly **500x** as the cluster size (k) increases from **5** to **50**. We see sklearn takes exponentially more time and a GPU can definitely speed up this process. Although the algorithms are not exactly the same, the performance and quality of results are comparable and the speedup is worth the minor loss in accuracy.

**PS:** _In all experiments, the runtime includes the time to load the image and cluster it._

Then we can perform a similarity search using these dominant colors to find similar images and by concatenating dominant colors we can find similar videos.
(I can explain)

### KNN similarity search

We can cluster smaller images/features and use the faiss - GPU implementation instead of sklearn.

**TODO**

### Using Wavelet image hash for similarity search:

https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5 - Uses [imagehash](https://pypi.org/project/ImageHash/), a python library to compute 1 of 4 different hashes and use hashes for comparison

**TODO**
