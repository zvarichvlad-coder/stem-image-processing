import os
import json
import csv
import cv2
from src.filters import to_gray, apply_gaussian, apply_sobel, otsu_threshold
from src.metrics import get_mse, get_psnr, get_ssim

def main():
    with open("config.json", "r") as f:
        cfg = json.load(f)

    raw_dir = cfg["paths"]["raw_dir"]
    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    results = []
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(valid_exts)]

    if not files:
        print("У папці data/raw не знайдено зображень!")
        return

    print(f"Знайдено зображень: {len(files)}. Починаємо обробку...")

    for fname in files:
        fpath = os.path.join(raw_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        orig_gray = to_gray(img)

        # Pipeline 1
        p1_s1 = orig_gray
        p1_s2 = apply_gaussian(p1_s1, cfg["gaussian"]["size"], cfg["gaussian"]["sigma"])
        p1_s3 = otsu_threshold(p1_s2)

        # Pipeline 2
        p2_s2 = apply_sobel(p1_s1)
        p2_s3 = otsu_threshold(p2_s2)

        base_name = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_dir, f"{base_name}_p1_s3.png"), p1_s3)
        cv2.imwrite(os.path.join(out_dir, f"{base_name}_p2_s3.png"), p2_s3)

        results.append({
            "filename": fname,
            "p1_mse": round(get_mse(orig_gray, p1_s3), 2),
            "p1_psnr": round(get_psnr(orig_gray, p1_s3), 2),
            "p1_ssim": round(get_ssim(orig_gray, p1_s3), 4),
            "p2_mse": round(get_mse(orig_gray, p2_s3), 2),
            "p2_psnr": round(get_psnr(orig_gray, p2_s3), 2),
            "p2_ssim": round(get_ssim(orig_gray, p2_s3), 4)
        })

    if results:
        with open(cfg["paths"]["metrics_file"], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("Успішно оброблено! Метрики збережено.")

if __name__ == "__main__":
    main()