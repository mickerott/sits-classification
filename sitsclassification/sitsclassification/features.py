import copy
import datetime
import random
import numpy as np
import pandas as pd
from itertools import product


def extract_random_pixels(patch_data, k):
    """
    Extract random pixels from image patch.
    :param patch_data: dictionary containing data as 4-D arrays, timestamps as 1-D array and metadata.
    :param k: number of pixels to select
    :return: dictionary containing data as 4-D arrays, timestamps as 1-D array and metadata.
    """
    # random selection of pixels by indexes
    random.seed(0)
    _, m, n, _ = patch_data['features'].shape
    index_tuples = list(product(range(m), range(n)))
    random.shuffle(index_tuples)

    # set up dict for outputs
    selected_tuples = (index_tuples[:k])
    selected_data = {'features': [], 'labels': []}
    if 'mask' in patch_data.keys():
        selected_data['mask'] = []
    selected_data['timestamps'] = patch_data['timestamps']
    selected_data['bbox'] = patch_data['bbox']

    # fill output dict with selected data
    for idx_tuple in selected_tuples:
        m = idx_tuple[0]
        n = idx_tuple[1]
        selected_data['features'].append(patch_data['features'][:, m, n, :][:, np.newaxis, np.newaxis, :])
        selected_data['labels'].append(patch_data['labels'][:, m, n, :][:, np.newaxis, np.newaxis, :])
        if 'mask' in patch_data.keys():
            selected_data['mask'].append(patch_data['mask'][:, m, n, :][:, np.newaxis, np.newaxis, :])
    selected_data['features'] = np.concatenate(selected_data['features'], axis=1)
    selected_data['labels'] = np.concatenate(selected_data['labels'], axis=1)
    if 'mask' in patch_data.keys():
        selected_data['mask'] = np.concatenate(selected_data['mask'], axis=1)

    return selected_data


def interpolate_features(selected_data, band_indexes=[0], interval=5, smooth=False):
    t, m, n, d = selected_data['features'].shape
    interpolated_features = []
    for i in range(m):
        selected_pixel = selected_data['features'][:, i, :, :]
        if 6 in band_indexes:
            red = selected_pixel[..., 2]
            nir = selected_pixel[..., 3]
            ndvi = (nir-red)/(nir+red)
            selected_pixel = np.concatenate([selected_pixel, ndvi[..., np.newaxis]], axis=-1)
        features = selected_pixel[..., band_indexes].reshape(t, len(band_indexes))
        interpolated_pixel, new_timestamps = interpolate_pixel(features, selected_data['timestamps'],
                                                               interval=interval, smooth=smooth,
                                                               winsize=interval*4, cutoff=0)
        interpolated_features.append(interpolated_pixel)
    interpolated_data = copy.copy(selected_data)
    interpolated_data['features'] = np.stack(interpolated_features, axis=1)[..., np.newaxis, :]
    interpolated_data['timestamps'] = new_timestamps
    return interpolated_data


def interpolate_pixel(features, timestamps, interval=1, smooth=False, winsize=None, cutoff=None):
    timestamps = np.array([datetime.datetime(2017, 1, 1) - datetime.timedelta(interval)] + list(timestamps) + [
        datetime.datetime(2017, 12, 31)], dtype=np.datetime64)  # TODO: hardcoded to 2017
    t, d = features.shape
    zeros = np.zeros((1, d), dtype='int')
    features = np.concatenate([zeros, features, zeros], axis=0)
    df = pd.DataFrame(features)
    df = pd.concat([df, pd.Series(timestamps, name='date')], axis=1)
    df.set_index('date', inplace=True)
    df = df.resample(f'{interval}D').mean().interpolate('linear')
    if smooth:
        winsize = int(np.rint(winsize / interval))
        cutoff_l = int(np.rint(cutoff / interval))
        cutoff_r = int(np.rint(cutoff / interval))
        df = df.rolling(winsize).mean()[cutoff_l+1:len(df)-cutoff_r+1]
    else:
        df = df.iloc[1:]
    interpolated_features = np.array(df)
    new_timestamps = list(df.index)
    return interpolated_features, new_timestamps
