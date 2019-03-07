import numpy as np
import h5py
import subprocess
import knn_cnn_features
import sys
sys.path.append("audioset/")
import vggish_inference

def extract_audio_from_video(vid_path):
    vid_name = vid_path.split('/')[-1].split('.')[0]
    p = subprocess.Popen("ffmpeg -loglevel panic -i "+vid_path+" -f wav -vn data/audio/"+vid_name+".wav",
                         stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()

def embedding_from_audio(wav_path, delete=True):
    _, audio_embedding = vggish_inference.main(wav_file=wav_path)
    if delete:
        p = subprocess.Popen("rm -r "+wav_path,
                             stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
    return audio_embedding

def load_sound_data(second_level=False,length=10):
    audioset_h5f = h5py.File('audioset_balanced_features_vggish.h5', 'r')
    audio_embeddings = np.array(audioset_h5f['audio_embeddings'],dtype='float32')
    audioset_h5f.close()
    true_labels = np.load('audioset_balanced_labels.npy')
    feature_labels = np.array([flab[0] for flab in true_labels])
    if second_level:
        merged_audio_embeddings = audio_embeddings.reshape(audio_embeddings.shape[:-3] + (-1,128))
        embedding_labels = np.array([i for i in feature_labels for _ in range(length)])
        true_labels = np.array([i for i in true_labels for _ in range(length)])
        return merged_audio_embeddings, embedding_labels, true_labels
    else:
        reshaped_audio_embeddings = audio_embeddings.reshape(audio_embeddings.shape[:-2] + (-1,))
        return reshaped_audio_embeddings, feature_labels, true_labels

def similar_sound_vids(vid_path, k=100):
    extract_audio_from_video(vid_path)
    audio_embedding = embedding_from_audio('data/audio/'+vid_path.split('/')[-1].split('.')[0]+'.wav')
    audioset_bal_embeddings, feature_labels, _ = load_sound_data()
    feature_indices = knn_cnn_features.run_knn_features(audioset_bal_embeddings,\
                                                    test_vectors=audio_embedding,k=k)
    return feature_labels[feature_indices]

def similar_sound_audio(wav_path, k=100):
    audio_embedding = embedding_from_audio(wav_path)
    audioset_bal_embeddings, feature_labels, _ = load_sound_data()
    feature_indices = knn_cnn_features.run_knn_features(audioset_bal_embeddings,\
                                                    test_vectors=audio_embedding,k=k)
    return feature_labels[feature_indices]

def similar_sound_embedding(audio_embedding, k=100):
    audioset_bal_embeddings, feature_labels, _ = load_sound_data()
    feature_indices = knn_cnn_features.run_knn_features(audioset_bal_embeddings,\
                                                    test_vectors=audio_embedding,k=k)
    return feature_labels[feature_indices]

def label_from_index(similar_indices, true_indices=None):
    import pandas as pd
    # true_labels = np.load('audioset_balanced_labels.npy')
    df = pd.read_csv('audioset/class_labels_indices.csv')
    similar_labels = df['display_name'].values[similar_indices]
    if type(true_indices) is np.ndarray:
        true_labels = df['display_name'].values[true_indices]
        return similar_labels, true_labels
    else:
        return similar_labels

def load_sound_data_ucf():
    feature_file = h5py.File('audio_sec_UCF_vggish.h5', 'r')
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

def similar_sound_ucf_video(vid_path, k=10, dist=False, verbose=False, newVid=False):
    if newVid:
        try:
            extract_audio_from_video(vid_path)
        except:
            print("No audio channel found")
        audio_embedding = embedding_from_audio('data/audio/'+vid_path.split('/')[-1].split('.')[0]+'.wav')
    else:
        audio_embedding = feature_vectors[np.where(feature_labels=="data/audio/"+vid_path.split('/')[-1].split(".")[-2]+".npy")]
    distances, feature_indices = knn_cnn_features.run_knn_features(feature_vectors,\
            test_vectors=feature_vectors[:10],k=k, dist=True, flat=True)
    # adjust for silence
    distances = [d+1000 for d in distances]
    del audio_embedding
    merged_similarities = multi_sec_inference(distances,feature_labels[feature_indices])
    if verbose:
        print(merged_similarities[1][:k])
    if dist:
        return merged_similarities[0][:k], list(map(str,merged_similarities[1][:k]))
    else:
        return merged_similarities[1][:k]

feature_vectors, feature_labels = load_sound_data_ucf()

# import time
# start = time.time()
# for i in range(5):
#     similar_sound_ucf_video('data/UCF101/v_ApplyEyeMakeup_g01_c01.webm', verbose=True, newVid=True)
# print((time.time()-start)/5)
# 2.0657553434371948 seconds (0.5 s if not new video)
