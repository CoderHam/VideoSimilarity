import numpy as np
import h5py
import subprocess
import knn_cnn_features
import sys
sys.path.append("audioset/")
import vggish_inference

def extract_audio_from_video(vid_path):
    vid_name = vid_path.split('/')[-1].split('.')[0]
    p = subprocess.Popen("ffmpeg -i "+vid_path+" -f wav -vn data/audio/"+vid_name+".wav",
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
