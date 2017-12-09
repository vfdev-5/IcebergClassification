
import numpy as np
import cv2


def center_crop(img, crop_size):
    h, w, c = img.shape
    center = (w//2, h//2)
    return img[center[1]-crop_size:center[1]+crop_size,center[0]-crop_size:center[0]+crop_size,:]


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


def segment_object_per_band(img):
    assert len(img.shape) == 3 and img.shape[2] == 2
    proc = smooth(img, sigmaX=0.7)    
    for i in range(2):
        b = proc[:, :, i]
        m1 = np.max(b[:20, :20])
        m2 = np.max(b[-20:, :20])
        m3 = np.max(b[-20:, -20:])
        m4 = np.max(b[20:, -20:])        
        t = np.mean([m1, m2, m3, m4])
        b[b < t] = -255.0
        b[b > t] = 255.0
        b[b < 0] = 0.0
    proc = proc.astype(np.uint8)
    return morpho_close(proc, ksize=5)   


def segment_object(img):
    proc = segment_object_per_band(img)
    proc = np.bitwise_and(proc[:, :, 0], proc[:, :, 1])
    return proc


def object_size(img):
    bin_img = segment_object(img)
    return np.count_nonzero(bin_img)
