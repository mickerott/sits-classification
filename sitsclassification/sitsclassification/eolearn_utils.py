from eolearn.core import EOPatch
import numpy as np
import gzip
import pickle
from pathlib import Path


def load_eopatch(file_path, square_subset=1e4, apply_mask=False):
    """
    Read data from eo-learn EOPatch.
    :param file_path:
    :param square_subset: subset upper left square of this edge length
    :return: pixels with known geographic positions
    """
    # subset coordinates
    mmin = 0
    mmax = square_subset
    nmin = 0
    nmax = square_subset

    # load data
    eopatch = EOPatch.load(Path(file_path), lazy_loading=True)
    features = eopatch['data']['BANDS'][:, mmin:mmax, nmin:nmax, :]
    valid_data = eopatch['mask']['VALID_DATA'][:, mmin:mmax, nmin:nmax, :]
    mask = ~valid_data.astype(np.bool)
    features = features.astype(np.float16)
    labels = eopatch['mask_timeless']['LULC'][mmin:mmax, nmin:nmax, :][np.newaxis, ...].astype(np.uint8)
    timestamps = eopatch['timestamp']
    timestamps = np.array(timestamps, dtype=np.datetime64)
    bbox = eopatch['bbox']
    bounds = bbox.geometry.bounds
    epsg = bbox.get_crs().epsg
    patch_data = {'features': features, 'labels': labels, 'timestamps': timestamps, 'bbox': {'bounds': bounds,
                                                                                             'epsg': epsg}}
    if apply_mask:
        features[np.repeat(mask, features.shape[-1], axis=-1)] = np.nan
    else:
        patch_data['mask'] = mask
    return patch_data

def load_eopatch_no_eolearn(file_path, square_subset=1e4):
    """TODO: to be completed"""
    timestamp_path = Path(file_path) / 'timestamp.pkl.gz'
    with gzip.open(timestamp_path, 'rb') as f:
        timestamps = pickle.load(f)
    bbox_path = Path(file_path) / 'bbox.pkl.gz'
    with gzip.open(bbox_path, 'rb') as f:
        bbox = pickle.load(f)
    features_path = Path(file_path) / 'data' / 'BANDS.npy.gz'
    f = gzip.GzipFile(features_path, 'r')
    features = np.load(f)
    features = features[:, :square_subset, :square_subset, :].astype(np.float16)
    print(features.shape)
    mask_path = Path(file_path) / 'mask' / 'VALID_DATA.npy.gz'
    f = gzip.GzipFile(mask_path, 'r')
    valid_data = np.load(f)
    mask = ~valid_data.astype(np.bool)
    print(valid_data.shape)
    return features, mask, timestamps, bbox
