import numpy as np
from tqdm import tqdm
import cv2

def detect_km_dist(km, x_i, y_i, per=.5):
    upper = np.max(km.shape[:2])
    lower = 1
    target_rgb = km[x_i, y_i, :3]
    if km[x_i, y_i, 3] == 0: return 0
    while ((x_i - lower >= 0 
            or  y_i - lower >= 0 
            or x_i + lower < km.shape[0] 
            or y_i + lower < km.shape[1]) 
           and not np.any(np.logical_and(np.mean(km[:, :, :3]
                      [x_i - lower if x_i - lower >= 0 else 0:
                       x_i + lower if x_i + lower < km.shape[0] else km.shape[0] - 1,
                       y_i - lower if y_i - lower >= 0 else 0:
                       y_i + lower if y_i + lower < km.shape[1] else km.shape[1] - 1] != target_rgb, axis=2), 
                                    km[:, :, 3]
                      [x_i - lower if x_i - lower >= 0 else 0:
                       x_i + lower if x_i + lower < km.shape[0] else km.shape[0] - 1,
                       y_i - lower if y_i - lower >= 0 else 0:
                       y_i + lower if y_i + lower < km.shape[1] else km.shape[1] - 1] == 255))):
        lower += int(np.ceil(5 * per))
    return lower

def kmeans_prob_dist(i_re_prob, names, k=0.1, per=.5):
    '''Calculate the probability distribution for each pixel from the kmeans'd edges in the image
    input:
        i_re_prob - element of the re_prob_padded
        names - names dict
    '''
    pbar = tqdm(list(i_re_prob['seg_re'].keys()), leave=False)
    for i in pbar:
        pbar.set_description("Processing %s's kmeans part" % names[i + 1])
        if i == -1:
            continue
        tmp_km = i_re_prob['seg_re'][i]['kmeans']
        original_size = tmp_km.shape[:2]
        tmp_km = cv2.resize(tmp_km, tuple(reversed(tuple(map(int, 
                                                 np.array(original_size) * per)))), interpolation=cv2.INTER_AREA)
        tmp_km_dist = np.ones(tmp_km.shape[:2])
        for x_i in range(tmp_km.shape[0]):
            for y_i in range(tmp_km.shape[1]):
                tmp_km_dist[x_i, y_i] = detect_km_dist(tmp_km, x_i, y_i)
        tmp_km_dist = cv2.resize(tmp_km_dist, tuple(reversed(original_size)), interpolation=cv2.INTER_AREA)
        i_re_prob['seg_re'][i].update({'km_dist': tmp_km_dist})
    return i_re_prob


def detect_edge_dist(km, x_i, y_i, per=.5):
    upper = np.max(km.shape[:2])
    lower = 1
    if km[x_i, y_i, 3] == 0:
        while ((x_i - lower > 0 
                or  y_i - lower > 0 
                or x_i + lower < km.shape[0] 
                or y_i + lower < km.shape[1]) 
               and np.all(km[:, :, 3]
                          [x_i - lower if x_i - lower > 0 else 0:
                           x_i + lower if x_i + lower < km.shape[0] else km.shape[0] - 1,
                           y_i - lower if y_i - lower > 0 else 0:
                           y_i + lower if y_i + lower < km.shape[1] else km.shape[1] - 1] == 0)):
            lower += int(np.ceil(5 * per))
        return - lower
    else:
        while ((x_i - lower > 0 
                or  y_i - lower > 0 
                or x_i + lower < km.shape[0] 
                or y_i + lower < km.shape[1]) 
               and np.all(km[:, :, 3]
                          [x_i - lower if x_i - lower > 0 else 0:
                           x_i + lower if x_i + lower < km.shape[0] else km.shape[0] - 1,
                           y_i - lower if y_i - lower > 0 else 0:
                           y_i + lower if y_i + lower < km.shape[1] else km.shape[1] - 1] != 0)):
            lower += int(np.ceil(5 * per))
        return lower

def edge_prob_dist(i_re_prob, names, k=0.1, per=.5):
    '''Calculate the probability distribution for each pixel from the edges in the image
    input:
        i_re_prob - element of the re_prob_padded
        names - names dict
    '''
    pbar = tqdm(list(i_re_prob['seg_re'].keys()), leave=False)
    for i in pbar:
        pbar.set_description("Processing %s's edge part" % names[i + 1])
        if i == -1:
            continue
        tmp_km = i_re_prob['seg_re'][i]['kmeans']
        original_size = tmp_km.shape[:2]
        tmp_km = cv2.resize(tmp_km, tuple(reversed(tuple(map(int, 
                                                 np.array(original_size) * per)))), interpolation=cv2.INTER_AREA)
        tmp_edge_dist = np.ones(tmp_km.shape[:2])
        for x_i in range(tmp_km.shape[0]):
            for y_i in range(tmp_km.shape[1]):
                tmp_edge_dist[x_i, y_i] = detect_edge_dist(tmp_km, x_i, y_i, per=per)
        tmp_edge_dist = cv2.resize(tmp_edge_dist, tuple(reversed(original_size)), interpolation=cv2.INTER_AREA)
        i_re_prob['seg_re'][i].update({'edge_dist': tmp_edge_dist})
    return i_re_prob