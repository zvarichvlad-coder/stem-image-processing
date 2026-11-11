"""
Модуль математичного розрахунку кількісних метрик порівняння якості зображень:
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)
"""
import numpy as np


def get_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Обчислює середньоквадратичну помилку (MSE) між двома зображеннями.

    :param img1: Перше 2D зображення.
    :param img2: Друге 2D зображення.
    :return: Числове значення MSE.
    """
    arr1 = img1.astype(np.float64)
    arr2 = img2.astype(np.float64)
    return float(np.mean((arr1 - arr2) ** 2))


def get_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Обчислює пікове відношення сигналу до шуму (PSNR) в децибелах (дБ).

    :param img1: Перше 2D зображення.
    :param img2: Друге 2D зображення.
    :return: Значення PSNR в децибелах (або 100.0 при повній тотожності).
    """
    mse = get_mse(img1, img2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    return float(10.0 * np.log10((max_pixel ** 2) / mse))


def get_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Обчислює індекс структурної подібності (SSIM) між двома зображеннями.

    :param img1: Оригінальне 2D зображення.
    :param img2: Оброблене 2D зображення.
    :return: Індекс SSIM у діапазоні [-1, 1].
    """
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)

    # Константи стабілізації за стандартом Ванга (Zhou Wang et al.)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    var_x = np.var(x)
    var_y = np.var(y)

    # Безпечне обчислення коваріації з обробкою суцільно однорідних матриць
    if x.size > 1 and (var_x > 0 or var_y > 0):
        cov_matrix = np.cov(x.flatten(), y.flatten())
        cov_xy = cov_matrix[0, 1] if cov_matrix.ndim == 2 else 0.0
    else:
        cov_xy = 0.0

    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)

    return float(num / den)