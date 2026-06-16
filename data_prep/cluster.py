"""
Training frame selection via k-means clustering.

For each video, encodes all face crop frames using a face recognition model,
clusters them with k-means, and selects the frame closest to each cluster
center as a training example. This ensures the training set covers the full
range of facial poses and expressions seen in the video.

The resulting cluster assignments are saved as a pickle file and used by the
DFD dataset class to split frames into training and inference sets.

Usage:
    python cluster.py
"""

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from pathlib import Path
from datetime import datetime
from glob import glob
import numpy as np
import face_recognition
import pickle

# ---------------------------------------------------------------------------
# Configuration — set these before running
# ---------------------------------------------------------------------------
ROOT = 'path/to/dataset'    # Root directory containing the face crops
KMEANS_CENTERS = 30         # Number of training frames to select per video
FRAMES = 5000               # Maximum number of frames to consider per video
# ---------------------------------------------------------------------------


def encode(path):
    """
    Compute face encodings for all images at the given glob path.

    Args:
        path: Glob pattern pointing to a set of face crop PNG files.

    Returns:
        List of dicts with keys 'fpath' (str) and 'enc' (128-d numpy array).
    """
    data = []
    for fpath in sorted(glob(path)):
        img = face_recognition.load_image_file(fpath)
        encoding = face_recognition.face_encodings(img, [(0, 255, 255, 0)])[0]
        data.append({'fpath': fpath, 'enc': encoding})
    return data


def cluster_data(data):
    """
    Run k-means on face encodings and return the index of the frame closest
    to each cluster center.

    Args:
        data: List of dicts as returned by encode().

    Returns:
        List of integer frame indices, one per cluster.
    """
    encodings = [d['enc'] for d in data]
    clt = KMeans(KMEANS_CENTERS, n_init=5)
    clt.fit(encodings)

    if clt.n_iter_ >= clt.max_iter:
        print('Warning: k-means may not have converged.')

    closest_points_idx, _ = pairwise_distances_argmin_min(clt.cluster_centers_, encodings)
    return closest_points_idx.tolist()


def main():
    """
    Encode and cluster frames for all videos, caching encodings to disk to
    avoid recomputing them on subsequent runs.
    """
    clusters = {}
    encodings_cache_path = f'{ROOT}/clusters_encodings_{FRAMES}.pck'
    encodings_dict = {}

    if Path(encodings_cache_path).exists():
        print('Using cached encodings.')
        with open(encodings_cache_path, 'rb') as f:
            encodings_dict = pickle.load(f)

    for folder in sorted(glob(f'{ROOT}/40/*')):
        vid_name = Path(folder).stem
        if vid_name.startswith('clusters'):
            continue

        if vid_name not in encodings_dict:
            encodings_dict[vid_name] = encode(f'{folder}/hq/*')

        clusters[vid_name] = cluster_data(encodings_dict[vid_name][:FRAMES])
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Finished: {vid_name}')

    with open(encodings_cache_path, 'wb') as f:
        pickle.dump(encodings_dict, f)

    with open(f'{ROOT}/clusters_{FRAMES}_{KMEANS_CENTERS}.pck', 'wb') as f:
        pickle.dump(clusters, f)


if __name__ == '__main__':
    main()
