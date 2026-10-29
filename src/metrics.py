import numpy as np

def get_mse(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

def get_psnr(img1, img2):
    mse = get_mse(img1, img2)
    return 100.0 if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)

def get_ssim(img1, img2):
    x, y = img1.astype(np.float64), img2.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    vxy = np.cov(x.flatten(), y.flatten())[0, 1]
    num = (2 * mx * my + c1) * (2 * vxy + c2)
    den = (mx**2 + my**2 + c1) * (vx + vy + c2)
    return float(num / den)