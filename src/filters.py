"""
Модуль низкорівневих алгоритмів обробки цифрових зображень.
Усі математичні операції реалізовано вручну з використанням NumPy
без залучення готових функцій обробки із спеціалізованих бібліотек CV.
"""

import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    Конвертує кольорове RGB/BGR зображення у півтонове (Grayscale).
    Застосовує зважену формулу яскравості (Luminosity Method):
    Y = 0.299*R + 0.587*G + 0.114*B

    :param img: Вхідне 3D масив-зображення (H, W, C) в форматі BGR.
    :return: 2D масив сірого кольору (H, W) типу uint8.
    """
    if img.ndim == 2:
        return img.astype(np.uint8)

    # OpenCV завантажує в форматі BGR (Channel 0: B, Channel 1: G, Channel 2: R)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def pad_image(img: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """
    Виконує граничне доповнення (Padding) для двовимірного масиву
    дублюванням крайніх пікселів (edge padding).

    :param img: Вхідне 2D зображення (H, W).
    :param pad_h: Ширина розширення по вертикалі.
    :param pad_w: Ширина розширення по горизонталі.
    :return: Доповнений масив розширеного розміру.
    """
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')


def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Універсальна векторна двовимірна дискретна згортка (2D Convolution).

    :param img: 2D вхідна матриця інтенсивностей (H, W).
    :param kernel: 2D матриця ядра згортки (kH, kW).
    :return: 2D результуюча матриця типу float64.
    """
    img_f = img.astype(np.float64)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = pad_image(img_f, pad_h, pad_w)
    h, w = img.shape
    output = np.zeros((h, w), dtype=np.float64)

    # Векторні зсуви для прискорення (без повільних 4-рівневих циклів Python)
    for i in range(kh):
        for j in range(kw):
            output += padded[i:i + h, j:j + w] * kernel[i, j]

    return output


def get_gaussian_kernel(size: int = 5, sigma: float = 1.2) -> np.ndarray:
    """
    Генерує нормалізоване 2D ядро Гауса заданого розміру та середньоквадратичного відхилення.

    :param size: Непарне ціле число (розмір ядра N x N).
    :param sigma: Параметр розсіювання Гауса.
    :return: Двовимірна нормована матриця розміром (size, size).
    """
    if size % 2 == 0:
        raise ValueError("Розмір ядра Гауса повинен бути непарним числом.")

    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def gaussian_blur(img: np.ndarray, size: int = 5, sigma: float = 1.2) -> np.ndarray:
    """
    Застосовує згладжуючий фільтр Гауса до півтонового зображення.

    :param img: 2D масив сірого кольору (H, W).
    :param size: Розмір ядра фільтра.
    :param sigma: Сигма розмиття.
    :return: Згладжене зображення (H, W) типу uint8.
    """
    kernel = get_gaussian_kernel(size, sigma)
    blurred = conv2d(img, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def sobel_filter(img: np.ndarray) -> np.ndarray:
    """
    Виділяє контури за допомогою диференціального оператора Собеля.
    Обчислює модулю градієнта інтенсивності: G = sqrt(Gx^2 + Gy^2).

    :param img: 2D масив сірого кольору (H, W).
    :return: Нормоване зображення контурів (H, W) типу uint8.
    """
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float64)

    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float64)

    gx = conv2d(img, kernel_x)
    gy = conv2d(img, kernel_y)

    magnitude = np.hypot(gx, gy)  # sqrt(gx^2 + gy^2)
    max_val = np.max(magnitude)

    if max_val > 0:
        magnitude = (magnitude / max_val) * 255.0

    return np.clip(magnitude, 0, 255).astype(np.uint8)


def otsu_threshold(img: np.ndarray) -> np.ndarray:
    """
    Автоматична порогова бінаризація за методом Оцу.
    Знаходить оптимальний поріг T, який максимізує міжкласову дисперсію.

    :param img: 2D масив сірого кольору (H, W).
    :return: Бінарне зображення (значення 0 або 255) типу uint8.
    """
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    total_pixels = img.size

    current_max = -1.0
    threshold = 0

    sum_total = np.dot(np.arange(256), hist)
    weight_background = 0.0
    sum_background = 0.0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        # Обчислення міжкласової дисперсії
        between_variance = (weight_background * weight_foreground *
                            ((mean_background - mean_foreground) ** 2))

        if between_variance > current_max:
            current_max = between_variance
            threshold = t

    binary = np.where(img >= threshold, 255, 0)
    return binary.astype(np.uint8)