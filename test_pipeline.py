"""
Модуль автоматичного unittest-тестування алгоритмів обробки зображень.
Перевіряє коректність виконання згортки, бінаризації та метрик (Критерій C3).
"""
import unittest
import numpy as np
from src.filters import to_gray, extract_green_channel, gaussian_blur, sobel_filter, otsu_threshold
from src.metrics import get_mse, get_psnr, get_ssim


class TestImageProcessingPipeline(unittest.TestCase):

    def test_to_gray_dimensions(self):
        """Перевірка правильності зменшення вимірності масиву при конвертації в сірий (Grayscale)."""
        dummy_rgb = np.ones((10, 10, 3), dtype=np.uint8) * 150
        gray = to_gray(dummy_rgb)
        self.assertEqual(gray.shape, (10, 10))
        self.assertEqual(gray[0, 0], 150)

    def test_green_channel_extraction(self):
        """Перевірка екстракції зеленого каналу (Green Channel)."""
        dummy_bgr = np.zeros((5, 5, 3), dtype=np.uint8)
        dummy_bgr[:, :, 1] = 180  # Канал index 1 = Green у форматі BGR
        green = extract_green_channel(dummy_bgr)
        self.assertEqual(green.shape, (5, 5))
        self.assertEqual(green[0, 0], 180)

    def test_sobel_filter_output(self):
        """Перевірка оператора Собеля (розмірність та діапазон значення [0, 255])."""
        dummy_img = np.array([[0, 0, 255, 255],
                              [0, 0, 255, 255],
                              [0, 0, 255, 255],
                              [0, 0, 255, 255]], dtype=np.uint8)
        edges = sobel_filter(dummy_img)
        self.assertEqual(edges.shape, (4, 4))
        self.assertTrue(np.max(edges) <= 255 and np.min(edges) >= 0)

    def test_otsu_binary_values(self):
        """Перевірка, що результат бінаризації Оцу містить виключно значення 0 або 255."""
        dummy_img = np.array([[10, 20, 30],
                              [200, 210, 220]], dtype=np.uint8)
        binary = otsu_threshold(dummy_img)
        unique_vals = set(np.unique(binary))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_metrics_identity(self):
        """Перевірка метрик на тотожних зображеннях (MSE=0, PSNR=100, SSIM=1)."""
        img = np.ones((20, 20), dtype=np.uint8) * 128
        self.assertEqual(get_mse(img, img), 0.0)
        self.assertEqual(get_psnr(img, img), 100.0)
        self.assertAlmostEqual(get_ssim(img, img), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()