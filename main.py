"""
Головний модуль запуску STEM-проєкту.
Забезпечує масову обробку 100+ зображень, генерацію результатів,
запис метрик у CSV та автоматичну візуалізацію графіків телеметрії.
"""

import os
import json
import csv
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.filters import to_gray, gaussian_blur, sobel_filter, otsu_threshold
from src.metrics import get_mse, get_psnr, get_ssim

# Налаштування логування в консоль
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pipeline_1(img: np.ndarray, cfg: dict):
    """Пайплайн 1: Grayscale -> Gaussian Blur -> Otsu Thresholding"""
    gray = to_gray(img)
    blurred = gaussian_blur(gray, size=cfg["gaussian"]["size"], sigma=cfg["gaussian"]["sigma"])
    binary = otsu_threshold(blurred)
    return gray, binary


def pipeline_2(img: np.ndarray, cfg: dict):
    """Пайплайн 2: Grayscale -> Sobel Edge Detection -> Otsu Thresholding"""
    gray = to_gray(img)
    edges = sobel_filter(gray)
    binary = otsu_threshold(edges)
    return gray, binary


def generate_plots(results: list, output_path: str):
    """Генерує та зберігає порівняльні аналітичні графіки метрик."""
    indices = range(1, len(results) + 1)

    p1_psnr = [r["p1_psnr"] for r in results]
    p2_psnr = [r["p2_psnr"] for r in results]

    p1_ssim = [r["p1_ssim"] for r in results]
    p2_ssim = [r["p2_ssim"] for r in results]

    plt.figure(figsize=(15, 6))

    # Графік PSNR
    plt.subplot(1, 2, 1)
    plt.plot(indices, p1_psnr, label="Пайплайн 1 (Гаус + Оцу)", color="#1f77b4", linewidth=1.5)
    plt.plot(indices, p2_psnr, label="Пайплайн 2 (Собель + Оцу)", color="#d62728", linewidth=1.5)
    plt.title("Розподіл метрики PSNR (дБ) по вибірці", fontsize=12, fontweight='bold')
    plt.xlabel("Номер зображення у вибірці", fontsize=10)
    plt.ylabel("PSNR (децибели)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Графік SSIM
    plt.subplot(1, 2, 2)
    plt.plot(indices, p1_ssim, label="Пайплайн 1 (Гаус + Оцу)", color="#1f77b4", linewidth=1.5)
    plt.plot(indices, p2_ssim, label="Пайплайн 2 (Собель + Оцу)", color="#d62728", linewidth=1.5)
    plt.title("Розподіл метрики SSIM по вибірці", fontsize=12, fontweight='bold')
    plt.xlabel("Номер зображення у вибірці", fontsize=10)
    plt.ylabel("Індекс SSIM", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Аналітичні графіки успішно збережено: {output_path}")


def main():
    cfg = load_config()
    raw_dir = cfg.get("raw_dir", "data/raw")
    proc_dir = cfg.get("proc_dir", "data/processed")
    os.makedirs(proc_dir, exist_ok=True)

    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(valid_exts)])

    if not files:
        logging.error(f"У папці {raw_dir} не знайдено жодного графічного файла!")
        return

    logging.info(f"Розпочато масову обробку вибірки з {len(files)} зображень...")

    results = []

    for idx, filename in enumerate(files, start=1):
        img_path = os.path.join(raw_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            logging.warning(f"Не вдалося зчитати файл: {filename}. Пропуск.")
            continue

        # Запуск двох конвеєрів
        g1, b1 = pipeline_1(img, cfg)
        g2, b2 = pipeline_2(img, cfg)

        # Збереження результатів обробки
        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(proc_dir, f"{base_name}_p1.png"), b1)
        cv2.imwrite(os.path.join(proc_dir, f"{base_name}_p2.png"), b2)

        # Розрахунок метрик відносно оригінального сірого кадру
        m1_mse, m1_psnr, m1_ssim = get_mse(g1, b1), get_psnr(g1, b1), get_ssim(g1, b1)
        m2_mse, m2_psnr, m2_ssim = get_mse(g2, b2), get_psnr(g2, b2), get_ssim(g2, b2)

        results.append({
            "id": idx,
            "filename": filename,
            "p1_mse": round(m1_mse, 2),
            "p1_psnr": round(m1_psnr, 2),
            "p1_ssim": round(m1_ssim, 4),
            "p2_mse": round(m2_mse, 2),
            "p2_psnr": round(m2_psnr, 2),
            "p2_ssim": round(m2_ssim, 4)
        })

        if idx % 10 == 0 or idx == len(files):
            logging.info(f"Опрацьовано [{idx}/{len(files)}] зображень.")

    # Запис підсумкового CSV звіту
    csv_path = cfg.get("csv_report", "metrics_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "filename", "p1_mse", "p1_psnr", "p1_ssim", "p2_mse", "p2_psnr", "p2_ssim"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logging.info(f"Метрики збережено у файл: {csv_path}")

    # Побудова графіків
    plot_path = cfg.get("plot_report", "data/processed/metrics_telemetry_plots.png")
    generate_plots(results, plot_path)
    logging.info("ОБРОБКУ ПОВНІСТЮ ЗАВЕРШЕНО УСПІШНО!")


if __name__ == "__main__":
    main()