import numpy as np
import h5py
import subprocess
import knn_cnn_features
import extract_features
import glob

# Using 8 since UCF has a median video length of less than 8 second
def extract_features_from_vid(vid_path, num_of_frames=8):
    p = subprocess.Popen("sh extractNFrames.sh "+vid_path+" "+str(num_of_frames), \
                     stdout=subprocess.PIPE, shell=True)
    p_status = p.wait()
    (output, err) = p.communicate()
    frame_list = glob.glob('data/tmp/*.jpg')
    vid_feature_vector = None
    for fr in frame_list:
        feature_vector = extract_features.get_vector_resnet50(fr)
        if type(vid_feature_vector) is not np.ndarray:
            vid_feature_vector = feature_vector
        else:
            vid_feature_vector = np.vstack((vid_feature_vector, feature_vector))
    return vid_feature_vector

def load_feature_data_ucf():
    feature_file = h5py.File('merged_features_UCF_resnet50.h5', 'r')
    feature_labels = np.array([fl.decode() for fl in feature_file['feature_labels']])
    feature_vectors = np.array(feature_file['feature_vectors'])
    feature_file.close()
    return feature_vectors, feature_labels

def get_ordered_unique(listed,dist):
    seen = set()
    seen_add = seen.add
    ordered_listed = [x for x in listed if not (x in seen or seen_add(x))]
    seen = set()
    seen_add = seen.add
    ordered_dist = [x for i, x in enumerate(dist) if not (listed[i] in seen or seen_add(listed[i]))]
    return ordered_listed, ordered_dist

def multi_sec_inference(distances, feature_indices):
    length = len(feature_indices)
    ordered_listed = []
    ordered_distances = []
    for i in range(length):
        ol, od = get_ordered_unique(feature_indices[i],distances[i])
        ordered_listed = ordered_listed + ol
        ordered_distances = ordered_distances + od
    # print(get_ordered_unique(ordered_listed, ordered_distances))
    sorted_listed = [x for _,x in sorted(zip(ordered_distances, ordered_listed))]
    uniq_sorted_listed, uniq_sorted_dist = get_ordered_unique(sorted_listed, sorted(ordered_distances))
    return uniq_sorted_dist, [usl.split('/')[-1].split('.')[0] for usl in uniq_sorted_listed]

def similar_feature_ucf_video(vid_path, k=10, dist=False, verbose=False, newVid=False):
    # feature_vectors, feature_labels = load_feature_data_ucf()
    if newVid:
        vid_feature_vector = extract_features_from_vid(vid_path)
    else:
        vid_feature_vector = feature_vectors[np.where(feature_labels=="data/UCF101/"+vid_path.split('/')[-1].split(".")[-2]+".avi")]
    distances, feature_indices = knn_cnn_features.run_knn_features(feature_vectors,\
        test_vectors=vid_feature_vector,k=k*4, dist=True, flat=True)
    # print(len(feature_indices), feature_labels[feature_indices])
    del vid_feature_vector
    merged_similarities = multi_sec_inference(distances,feature_labels[feature_indices])
    if verbose:
        print(merged_similarities[1][:k])
    if dist:
        return merged_similarities[0][:k], merged_similarities[1][:k]
    else:
        return merged_similarities[1][:k]

# load only once
feature_vectors, feature_labels = load_feature_data_ucf()

# import time
# start = time.time()
# for i in range(5):
#     similar_feature_ucf_video('data/UCF101/v_ApplyEyeMakeup_g01_c01.webm',verbose=True, newVid=False)
# print((time.time()-start)/5)
# 0.7 seconds (0.3 s if no not video)
