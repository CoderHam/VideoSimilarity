import color_similarity_inference
import sound_similarity_inference
import feature_similarity_inference
import cnn3d_similarity_inference

def get_ordered_unique(listed,dist):
    seen = set()
    seen_add = seen.add
    ordered_listed = [x for x in listed if not (x in seen or seen_add(x))]
    seen = set()
    seen_add = seen.add
    ordered_dist = [x for i, x in enumerate(dist) if not (listed[i] in seen or seen_add(listed[i]))]
    return ordered_listed, ordered_dist

def merge_similarity_ucf_video(vid_path, k=10, verbose=True):
    color_dist, color_indices = color_similarity_inference.similar_color_ucf_video(vid_path, k=k, dist=True)
    feature_dist, feature_indices = feature_similarity_inference.similar_feature_ucf_video(vid_path, k=k, dist=True)
    sound_dist, sound_indices = sound_similarity_inference.similar_sound_ucf_video(vid_path, k=k, dist=True)
    cnn3d_dist, cnn3d_indices = cnn3d_similarity_inference.similar_cnn3d_ucf_video(vid_path, k=k, dist=True)
    sorted_listed = [x for _,x in sorted(zip(color_dist+feature_dist+sound_dist+cnn3d_dist, color_indices+feature_indices+sound_indices+cnn3d_indices))]
    uniq_sorted_listed, uniq_sorted_dist = get_ordered_unique(sorted_listed, sorted(color_dist+feature_dist+sound_dist+cnn3d_dist))
    if verbose:
        print(uniq_sorted_listed[:k], uniq_sorted_dist[:k])
    return uniq_sorted_listed[:k]

# import time
# start = time.time()
# # # for i in range(10):
# merge_similarity_ucf_video('data/UCF101/v_ApplyEyeMakeup_g01_c01.avi', verbose=True)
# print((time.time()-start)/10)
#  3.8777024269104006 seconds
