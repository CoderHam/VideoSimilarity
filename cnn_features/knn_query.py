"""
This script queries the extracted CNN features for closest frame using
Facebook's FAISS algorithm.

It then outputs the query results along with the query image.
"""

# imports
import os
import time
import numpy as np
import pickle
import h5py
import faiss

class knn_query():
    """
    Class for kNN query using FAISS
    """

    def __init__(self, dims, xb, labels):
        """ Constructor
            Args:
                dims = number of dimensions
                index = FAISS index object
                xb = numpy array of feature vectors
                labels = dict, index to frame path mapping
                isTrained = boolean, is the index trained?
        """
        self.dims = dims
        self.index = None
        self.xb = xb
        self.labels = labels
        self.isTrained = False

    def train_knn_index(self):
        """
        This function trains the index using the extracted features so that
        kNN query can be performed. It just needs to be run with the extracted
        features.
        """
        # build the index
        s_time = time.time()
        self.index = faiss.IndexFlatL2(self.dims)
        self.isTrained = True
        # add feature vectors to the index
        print('Adding feature vectors to the index...')
        self.index.add(self.xb)
        e_time = time.time()
        index_time = round(e_time - s_time, 3)
        print('{} feature vectors added'.format(self.index.ntotal))
        print('Index finished in {}s'.format(index_time))

    def perform_query(self, xq, k):
        """
        This method performs the kNN query on trained index of the
        FAISS index object.
        """
        if not self.isTrained:
            print('You must train the index first!')
        else:
            # perform the query
            print('Performing {} nearest neighbor search...'.format(k))
            s_time = time.time()
            D, I = self.index.search(x=xq, k=k)
            e_time = time.time()
            query_time = round(e_time - s_time, 3)
            return D, I, query_time


def output_results(I, labels, query_image, query_time):
    print('\nQuery Completed in {}s'.format(query_time))
    print('\nQuery Frame:\n', query_image)
    print('\n')
    print('Top Similar Frames:')
    for i, ind in enumerate(I[0]):
        print(i+1, ': ', labels[ind])


def main():
    """
    These are the query settings. Adjust as necessary to perform the
    query.
    """
    # paths and naming conventions
    data_path = '/mnt/e/ucf_101/'

    # read file
    with open('features.pickle', 'rb') as handle:
        xb, ind2path_labels, path2ind_labels = pickle.load(handle)

    # get query frame feature vector
    query_image = data_path + 'v_ApplyEyeMakeup_g03_c01.avi/output000001.png'
    # query_image = 'v_Kayaking_g17_c06.avi/output000001.png'
    xq_ind = path2ind_labels[query_image]
    xq = np.reshape(xb[xq_ind, :], (1, -1))

    # initialize knn_query object and train index
    d = 512
    query = knn_query(d, xb, ind2path_labels)
    query.train_knn_index()

    # perform query and output results
    k = 15
    D, I, query_time = query.perform_query(xq=xq, k=k)
    output_results(I, ind2path_labels, query_image, query_time)


if __name__ == '__main__':
    main()
