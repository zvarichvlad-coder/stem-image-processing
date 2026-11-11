import os
import json
import csv
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.filters import to_gray, extract_green_channel, gaussian_blur, sobel_filter, otsu_threshold
from src.metrics import get_mse, get_psnr, get_ssim

# Налаштування логування в консоль
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def imread_unicode(path: str) -> np.ndarray:
    """Зчитує зображення з підтримкою кирилиці та юнікоду в шляхах файлів."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_config(path: str = "config.json") -> dict:
    """Завантажує налаштування з JSON-файлу або повертає дефолтні."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "gaussian": {"size": 5, "sigma": 1.2},
        "raw_dir": "data/raw",
        "proc_dir": "data/processed",
        "csv_report": "metrics_report.csv",
        "plot_report": "data/processed/metrics_telemetry_plots.png"
    }


def pipeline_1(img: np.ndarray, cfg: dict):
    """
    Пайплайн 1 (3 етапи):
    1. Grayscale (зважений напівтоновий простір)
    2. Gaussian Blur (згладжування шуму)
    3. Otsu Thresholding (порогова сегментація)
    """
    g_size = cfg.get("gaussian", {}).get("size", 5)
    g_sigma = cfg.get("gaussian", {}).get("sigma", 1.2)
    
    gray = to_gray(img)
    blurred = gaussian_blur(gray, size=g_size, sigma=g_sigma)
    binary = otsu_threshold(blurred)
    return gray, binary


def pipeline_2(img: np.ndarray, cfg: dict):
    """
    Пайплайн 2 (3 етапи):
    1. Green Channel (екстракція зеленого каналу)
    2. Sobel Edge Detection (виділення контурів)
    3. Otsu Thresholding (порогова сегментація)
    """
    green = extract_green_channel(img)
    edges = sobel_filter(green)
    binary = otsu_threshold(edges)
    return green, binary


def print_pretty_table(results: list):
    """Виводить відформатовану таблицю метрик у консоль."""
    if not results:
        return

    headers = ["ID", "Файл", "P1 MSE", "P1 PSNR", "P1 SSIM", "P2 MSE", "P2 PSNR", "P2 SSIM"]
    
    max_fn_len = max(len(r["filename"]) for r in results)
    col_fn_width = max(max_fn_len + 2, 12)

    col_widths = {
        "id": 6,
        "filename": col_fn_width,
        "p1_mse": 11, "p1_psnr": 10, "p1_ssim": 10,
        "p2_mse": 11, "p2_psnr": 10, "p2_ssim": 10
    }

    top_line = f"┌{'─'*col_widths['id']}┬{'─'*col_widths['filename']}┬{'─'*col_widths['p1_mse']}┬{'─'*col_widths['p1_psnr']}┬{'─'*col_widths['p1_ssim']}┬{'─'*col_widths['p2_mse']}┬{'─'*col_widths['p2_psnr']}┬{'─'*col_widths['p2_ssim']}┐"
    sep_line = f"├{'─'*col_widths['id']}┼{'─'*col_widths['filename']}┼{'─'*col_widths['p1_mse']}┼{'─'*col_widths['p1_psnr']}┼{'─'*col_widths['p1_ssim']}┼{'─'*col_widths['p2_mse']}┼{'─'*col_widths['p2_psnr']}┼{'─'*col_widths['p2_ssim']}┤"
    bot_line = f"└{'─'*col_widths['id']}┴{'─'*col_widths['filename']}┴{'─'*col_widths['p1_mse']}┴{'─'*col_widths['p1_psnr']}┴{'─'*col_widths['p1_ssim']}┴{'─'*col_widths['p2_mse']}┴{'─'*col_widths['p2_psnr']}┴{'─'*col_widths['p2_ssim']}┘"

    print("\n" + "=" * 90)
    print("                         ПІДСУМКОВА ТАБЛИЦЯ МЕТРИК ЯКОСТІ")
    print("=" * 90)
    print(top_line)
    
    print(f"│{headers[0]:^{col_widths['id']}}│"
          f"{headers[1]:^{col_widths['filename']}}│"
          f"{headers[2]:^{col_widths['p1_mse']}}│"
          f"{headers[3]:^{col_widths['p1_psnr']}}│"
          f"{headers[4]:^{col_widths['p1_ssim']}}│"
          f"{headers[5]:^{col_widths['p2_mse']}}│"
          f"{headers[6]:^{col_widths['p2_psnr']}}│"
          f"{headers[7]:^{col_widths['p2_ssim']}}│")
    print(sep_line)

    for r in results:
        fname = r['filename']
        if len(fname) > col_widths['filename'] - 2:
            fname = fname[:col_widths['filename'] - 5] + "..."

        print(f"│{r['id']:^{col_widths['id']}}│ "
              f"{fname:<{col_widths['filename']-1}}│"
              f"{r['p1_mse']:>{col_widths['p1_mse']-1}.2f} │"
              f"{r['p1_psnr']:>{col_widths['p1_psnr']-1}.2f} │"
              f"{r['p1_ssim']:>{col_widths['p1_ssim']-1}.4f} │"
              f"{r['p2_mse']:>{col_widths['p2_mse']-1}.2f} │"
              f"{r['p2_psnr']:>{col_widths['p2_psnr']-1}.2f} │"
              f"{r['p2_ssim']:>{col_widths['p2_ssim']-1}.4f} │")

    print(bot_line + "\n")


def generate_plots(results: list, output_path: str):
    """Генерує та зберігає порівняльні аналітичні графіки метрик."""
    if not results:
        return

    indices = range(1, len(results) + 1)

    p1_psnr = [r["p1_psnr"] for r in results]
    p2_psnr = [r["p2_psnr"] for r in results]

    p1_ssim = [r["p1_ssim"] for r in results]
    p2_ssim = [r["p2_ssim"] for r in results]

    plt.figure(figsize=(15, 6))

    # Графік PSNR
    plt.subplot(1, 2, 1)
    plt.plot(indices, p1_psnr, label="Пайплайн 1 (Grayscale + Гаус + Оцу)", color="#1f77b4", linewidth=1.5)
    plt.plot(indices, p2_psnr, label="Пайплайн 2 (Green + Собель + Оцу)", color="#d62728", linewidth=1.5)
    plt.title("Розподіл метрики PSNR (дБ) по вибірці", fontsize=12, fontweight='bold')
    plt.xlabel("Номер зображення у вибірці", fontsize=10)
    plt.ylabel("PSNR (децибели)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Графік SSIM
    plt.subplot(1, 2, 2)
    plt.plot(indices, p1_ssim, label="Пайплайн 1 (Grayscale + Гаус + Оцу)", color="#1f77b4", linewidth=1.5)
    plt.plot(indices, p2_ssim, label="Пайплайн 2 (Green + Собель + Оцу)", color="#d62728", linewidth=1.5)
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
    if not os.path.exists(raw_dir):
        logging.error(f"Папку {raw_dir} не знайдено!")
        return

    files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(valid_exts)])

    if not files:
        logging.error(f"У папці {raw_dir} не знайдено жодного графічного файла!")
        return

    logging.info(f"Розпочато масову обробку вибірки з {len(files)} зображень...")

    results = []
    csv_rows = []

    for idx, filename in enumerate(files, start=1):
        img_path = os.path.join(raw_dir, filename)
        img = imread_unicode(img_path)

        if img is None:
            logging.warning(f"Не вдалося зчитати файл: {filename}. Пропуск.")
            continue

        # Виконання двох пайплайнів
        g1, b1 = pipeline_1(img, cfg)
        g2, b2 = pipeline_2(img, cfg)

        base_name = os.path.splitext(filename)[0]
        
        # Збереження оброблених фото у папку data/processed/ із суфіксами _pipeline_1 та _pipeline_2
        cv2.imencode('.png', b1)[1].tofile(os.path.join(proc_dir, f"{base_name}_pipeline_1.png"))
        cv2.imencode('.png', b2)[1].tofile(os.path.join(proc_dir, f"{base_name}_pipeline_2.png"))

        # Обчислення метрик якості
        m1_mse, m1_psnr, m1_ssim = get_mse(g1, b1), get_psnr(g1, b1), get_ssim(g1, b1)
        m2_mse, m2_psnr, m2_ssim = get_mse(g2, b2), get_psnr(g2, b2), get_ssim(g2, b2)

        res_dict = {
            "id": idx,
            "filename": filename,
            "p1_mse": round(m1_mse, 2),
            "p1_psnr": round(m1_psnr, 2),
            "p1_ssim": round(m1_ssim, 4),
            "p2_mse": round(m2_mse, 2),
            "p2_psnr": round(m2_psnr, 2),
            "p2_ssim": round(m2_ssim, 4)
        }
        results.append(res_dict)

        # Форматування чисел для Excel з комою замість крапки
        csv_rows.append({
            "id": idx,
            "filename": filename,
            "p1_mse": str(round(m1_mse, 2)).replace(".", ","),
            "p1_psnr": str(round(m1_psnr, 2)).replace(".", ","),
            "p1_ssim": str(round(m1_ssim, 4)).replace(".", ","),
            "p2_mse": str(round(m2_mse, 2)).replace(".", ","),
            "p2_psnr": str(round(m2_psnr, 2)).replace(".", ","),
            "p2_ssim": str(round(m2_ssim, 4)).replace(".", ",")
        })

        if idx % 15 == 0 or idx == len(files):
            logging.info(f"Опрацьовано [{idx}/{len(files)}] зображень.")

    # Запис CSV звітів
    csv_path = cfg.get("csv_report", "metrics_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["id", "filename", "p1_mse", "p1_psnr", "p1_ssim", "p2_mse", "p2_psnr", "p2_ssim"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(csv_rows)

    logging.info(f"Метрики збережено у файл: {csv_path}")

    plot_path = cfg.get("plot_report", "data/processed/metrics_telemetry_plots.png")
    generate_plots(results, plot_path)
    print_pretty_table(results)

    logging.info("ОБРОБКУ ПОВНІСТЮ ЗАВЕРШЕНО УСПІШНО!")


if __name__ == "__main__":
    main()