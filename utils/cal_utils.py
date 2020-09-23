import numpy as np
import cv2

def calc_prob(pixel_dist, farthest_dist, nearest_prob, farthest_prob, nearest_dist=1):
    '''Calculate the prob of this pixel along the linear distribution / distance
    Input:
        pixel original value - the input value of pixel(x_t)
        farthest_dist - x1
        nearest_dist - x0
        nearest_prob - y0
        farthest_prob - y1
    Output:
        y_t =  y1 + (x_t - x1)[(y0 - y1)/(x0 - x1)]
    '''
    k = (farthest_prob - nearest_prob) / (farthest_dist - nearest_dist)
    y = nearest_prob + k * (pixel_dist - nearest_dist)
    return y

def redefine_prob(original_img, seg_re, overlay, edge_nearest_prob=1,
                edge_farthest_prob=0.2, km_nearest_prob=1, km_farthest_prob=0.2,
                overlay_alpha=1):
    '''Calculate and redefine the probablity of each pixel

    KMeans: ||---->|<--->|<-----||

            F ___/ N \_/ N \____ F
        or
            F \___ N ___ N ____/ F

    Edges:  ||<---------------->||

            N \_______F________/ N
        or
            N _______/F\________ N
    
    Overlay:
            ___/----\____/\____ * Alpha

    Input:
        original_img
        seg_re
        overlay
        edge_nearest_prob - The probability of this pixel if its distance from edge is '1'
        edge_farthest_prob - The probability of this pixel if its distance from edge is the maximum
    Output:
        the redefined probability
    Example:
        plt.imshow(redefine_prob(re_prob_padded[0]['original_img'],
            re_prob_padded[0]['seg_re'],
            re_prob_padded[0]['overlay'],
                        edge_nearest_prob=0.0,
                        edge_farthest_prob=.5,
                        km_nearest_prob=.5,
                        km_farthest_prob=0.0,
                        overlay_alpha=3))
    '''
    tmp_combine = np.zeros(original_img.shape[:2], dtype='float64')
    for i in seg_re:
        if i == -1:
            continue
        xy = seg_re[i]['xy']
        edge = seg_re[i]['edge_dist'].copy()
        km = seg_re[i]['km_dist'].copy()
        edge_prob = calc_prob(edge, edge.max(),
                            nearest_prob=edge_nearest_prob,
                            farthest_prob=edge_farthest_prob)
        km_prob = calc_prob(km, km.max(),
                            nearest_prob=km_nearest_prob,
                            farthest_prob=km_farthest_prob)
        if (edge_nearest_prob > edge_farthest_prob):
            edge_prob[edge_prob > edge_nearest_prob] = 0
        else:
            edge_prob[edge_prob < edge_nearest_prob] = 0
        if (km_nearest_prob > km_farthest_prob):
            km_prob[km_prob > km_nearest_prob] = 0
        else:
            km_prob[km_prob < km_nearest_prob] = 0
        tmp_combine[xy[0] : xy[1],
                    xy[2] : xy[3]] += edge_prob + km_prob
    tmp_combine[5:-5, 5:-5] = tmp_combine[5:-5, 5:-5] + overlay[0] * overlay_alpha
    return tmp_combine


def redefine_prob_list(re_prob_padded, edge_nearest_prob=0.0, edge_farthest_prob=1.0,
                       km_nearest_prob=1.0, km_farthest_prob=0.0, overlay_alpha=3):
    '''Make a list out of all images probability redefined
    
    Input:
        re_prob_padded
        edge_nearest_prob
        edge_farthest_prob
        km_nearest_prob
        km_farthest_prob
        overlay_alpha
    Output:
        list - overlay result
    '''
    re_prob_list = []
    for i_re_prob in re_prob_padded:
        re_prob_list.append(redefine_prob(
             i_re_prob['original_img'],
             i_re_prob['seg_re'],
             i_re_prob['overlay'],
            edge_nearest_prob=edge_nearest_prob,
            edge_farthest_prob=edge_farthest_prob,
            km_nearest_prob=km_nearest_prob,
            km_farthest_prob=km_farthest_prob,
            overlay_alpha=overlay_alpha
        ))
    return re_prob_list

def prob_xy_limit(x_raw, y_raw, width, limit_shape):
    '''Limit the boundry we find
    '''
    x_l = int(x_raw - width / 2) if x_raw - width / 2 >= 0 else 0
    x_r = int(x_raw + width / 2) if x_raw + width / 2 < limit_shape[0] else limit_shape[0] - 1
    y_u = int(y_raw - width / 2) if y_raw - width / 2 >= 0 else 0
    y_l = int(y_raw + width / 2) if y_raw + width / 2 < limit_shape[1] else limit_shape[1] - 1
    return x_l, x_r, y_u, y_l

def calc_img_weighted_distance(target, prob, max_prob=None, canvas=None):
    '''Calculate the weighted L2 distance between target image part and current canvas part
    Input:
        target - target image part (H * W * 4 or H * W * 3)
        prob - part of probability distribution (H1 * W1)
        max_prob - the pixels' probability maximum in the whole image (float)
        canvas - canvas part (H * W * 4 or H * W * 3) 
                    or None - initialize a white canvas (H * W * 3) 
    Output:
        Mean RGB distance of all pixels
    '''
    if canvas is None:
        dis_mat = ((target[:, :, :3] - np.full(target[:, :, :3].shape, 255)) / 255) ** 2
    else:
        dis_mat = ((target[:, :, :3] - canvas[:, :, :3]) / 255) ** 2
    if max_prob is None: max_prob = np.max(prob)
    return np.mean((dis_mat.mean(2) * np.exp(prob / max_prob))[target[:, :, 3] != 0.])

def generate_xy_dis_by_prob(prob_dis, step, per=.25):
    '''Generate the random positions following the given distribution
    Input:
        prob_dis - the probability distribution we wanna follow
        step - how many dots we generate per time (keep it proper)
        per - how small we resize the original probability distribution so that we can make this part efficent
    Output:
        [(x, y) ...] - List of the position we generated
    '''
    score_mat = np.ravel(cv2.resize(prob_dis, (int(prob_dis.shape[1] * per),
                                        int(prob_dis.shape[0] * per))))
    score_hat = score_mat + np.random.gumbel(0, 1, size=(step,) + score_mat.shape)
    return [(int((i / (prob_dis.shape[1] * per))/ per), int((i % (prob_dis.shape[1] * per)) / per)) for i in score_hat.argmax(-1)]

def crop_part(x_l, x_r, y_u, y_l, tmp_canvas, tmp_target, tmp_prob=None):
    '''Crop the all canvas / target / prob parts out of their parents
    '''
    tmp_canvas_part = tmp_canvas[x_l : x_r, y_u : y_l].copy()
    tmp_target_part = tmp_target[x_l : x_r, y_u : y_l].copy()
    tmp_prob_part = None
    if tmp_prob is not None:
        tmp_prob_part = tmp_prob[x_l : x_r, y_u : y_l].copy()
    return tmp_canvas_part, tmp_target_part, tmp_prob_part

def smooth(img, divide, width):
    def smooth_pix(img, tx, ty):
        if tx == divide * width - 1 or ty == divide * width - 1 or tx == 0 or ty == 0: 
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(divide):
        for q in range(divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img
