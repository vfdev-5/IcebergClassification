
import numpy as np
import cv2


def Scharr(img, ddepth, dx, dy):
    h, w, c = img.shape
    res = np.zeros_like(img)
    for i in range(c):
        res[:, :, i] = cv2.Scharr(img[:, :, i], ddepth, dx, dy)
    return res


def smooth(img, ksize=(5, 5), sigmaX=0.75):
    proc = cv2.GaussianBlur(img, ksize=ksize, sigmaX=sigmaX)
    return proc


def derivative(img):
    proc_x = Scharr(img, cv2.CV_32F, 1, 0)
    proc_y = Scharr(img, cv2.CV_32F, 0, 1)
    proc = cv2.magnitude(proc_x, proc_y)  
    return proc


def contours(img, t1, t2):
    h, w, c = img.shape
    res = np.zeros_like(img)
    for i in range(c):
        res[:, :, i] = cv2.Canny(img[:, :, i], t1, t2)
    return res
    

def threshold(img, t):
    proc = img.copy()
    for i in range(2):
        b = proc[:, :, i]
        b[b < t] = -255
        b[b > t] = 255
        b[b < 0] = 0
    return proc


def erode(img, ksize):
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.erode(img, kernel)


def dilate(img, ksize):
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.dilate(img, kernel)


def morpho_open(img, ksize):
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    
def morpho_close(img, ksize):
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def sea_4_corners_threshold(b, size=20):
    m1 = np.max(b[:size, :size])
    m2 = np.max(b[-size:, :size])
    m3 = np.max(b[-size:, -size:])
    m4 = np.max(b[size:, -size:])
    t = np.percentile([m1, m2, m3, m4], q=80)
    b[b < t] = -255.0
    b[b > t] = 255.0
    b[b < 0] = 0.0
    return b


def segment_object_per_band(img):
    assert len(img.shape) == 3 and img.shape[2] == 2
    proc = smooth(img, sigmaX=0.7)    
    for i in range(2):
        proc[:, :, i] = sea_4_corners_threshold(proc[:, :, i])
    proc = proc.astype(np.uint8)
    return morpho_close(proc, ksize=5)   


def segment_object_(img):
    proc = segment_object_per_band(img)
    proc = np.bitwise_and(proc[:, :, 0], proc[:, :, 1])
    return proc


def segment_object(img):
    assert len(img.shape) == 3 and img.shape[2] == 2
    proc = img[:, :, 0] + img[:, :, 1]
    proc = smooth(proc, sigmaX=0.7)
    proc = sea_4_corners_threshold(proc, size=25)
    proc = proc.astype(np.uint8)
    return proc


def object_size(img):
    bin_img = segment_object(img)
    return np.count_nonzero(bin_img)


def object_size_(img):
    bin_img = segment_object_(img)
    return np.count_nonzero(bin_img)


def center_crop(img, size):
    h, w, _ = img.shape
    th, tw = size, size
    if w == tw and h == th:
        params = (0, 0, h, w)
    else:
        i = (h - th) // 2
        j = (w - tw) // 2
        params = (i, j, th, tw)
    i, j, h, w = params
    return img[i:i + h, j:j + w, :]


def smart_crop(img, size):
    h, w, _ = img.shape
    ret = segment_object(img)
    m = np.sum(ret)
    x = np.arange(w)[None, :]
    y = np.arange(h)[:, None]
    px = int(np.sum(x * ret) / m) if m > 0 else 75 // 2
    py = int(np.sum(y * ret) / m) if m > 0 else 75 // 2
    th, tw = size, size
    if w == tw and h == th:
        params = (0, 0, h, w)
    else:
        params = (min(max(py - th // 2, 0), h - th),
                  min(max(px - tw // 2, 0), w - tw),
                  th, tw)
    i, j, h, w = params
    return img[i:i + h, j:j + w, :]
