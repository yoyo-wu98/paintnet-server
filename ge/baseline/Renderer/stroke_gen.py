import cv2
import numpy as np
from lagrange import lagrange

def normal(x, width):
    return (int)(x * (width - 1) + 0.5)


def draw(f, width=128, wl_dict=None, k=None):
    '''Render the output of fcn (10,) -> (width, width, ) : stroke weight
    Input:
        f - Output of the fcn
        wl_dict - lagrange interpolation input
                    or None for remain original
        k - linear decrease (expect k ~ [0, 1])
                or None for remain original 
    '''
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f.detach().numpy()
    if wl_dict is not None:
        wl_dict.update({0 : 0})
        w0 = lagrange(wl_dict, w0)
        w2 = lagrange(wl_dict, w2)
    elif k is not None:
        w0 = 1 - (1 - w0) * k
        w2 = 1 - (1 - w2) * k
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, width * 2]).astype('float32')
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(width, width))
