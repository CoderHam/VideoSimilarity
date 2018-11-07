from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_dom_colors(image_path,clusters=10,plot=False):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# flatten from 3d to 2d 
	image = image.reshape((image.shape[0] * image.shape[1], 3))
	# cluster the pixel intensities
	clt = KMeans(n_clusters = clusters)
	clt.fit(image)

	dcolors = [c.astype("uint8").tolist() for c in clt.cluster_centers_]

	if plot:
		hist = centroid_histogram(clt)
		bar = plot_colors(hist, clt.cluster_centers_)
		plt.figure()
		plt.axis("off")
		plt.imshow(bar)
		plt.show()

	return dcolors


def plot_colors(hist, centroids):

	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	for (percent, color) in zip(hist, centroids):
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX

	return bar

def centroid_histogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins=numLabels)

	hist = hist.astype("float")
	hist /= hist.sum()

	return hist

# import time
# t0 = time.time()
# get_dom_colors("data/images/golden1.jpg",clusters=50)
# t1 = time.time()
# print("k-Means runtime: %.3f s" % (t1 - t0))
