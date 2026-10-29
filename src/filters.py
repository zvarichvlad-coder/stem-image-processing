import numpy as np

def to_gray(img):
    if len(img.shape) == 2:
        return img
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)

def conv2d(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    res = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            res[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return np.clip(res, 0, 255).astype(np.uint8)

def apply_gaussian(img, size=5, sigma=1.0):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return conv2d(img, kernel)

def apply_sobel(img):
    gray = to_gray(img)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    gx = conv2d(gray, kx).astype(np.float64)
    gy = conv2d(gray, ky).astype(np.float64)
    g = np.sqrt(gx**2 + gy**2)
    return ((g / g.max()) * 255 if g.max() > 0 else g).astype(np.uint8)

def otsu_threshold(img):
    gray = to_gray(img)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    best_thresh, max_var = 0, 0
    sum_all = np.sum(np.arange(256) * hist)
    sum_b, w_b = 0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * ((m_b - m_f) ** 2)
        if var_between > max_var:
            max_var = var_between
            best_thresh = t
    out = np.zeros_like(gray)
    out[gray > best_thresh] = 255
    return out