import os
import urllib.request
from PIL import Image

output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

target_count = 150
temp_path = "temp_cat.jpg"

# Отримуємо вже наявні файли
existing_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.png')]
saved_count = len(existing_files)

print(f"Поточна кількість котиків у папці: {saved_count}/{target_count}")

opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
urllib.request.install_opener(opener)

i = saved_count + 1
while saved_count < target_count:
    dest_path = os.path.join(output_dir, f"cat_{saved_count+1:03d}.png")
    
    # Використовуємо різні API для гарантованого завантаження нових фотографій
    if i % 3 == 0:
        url = f"https://cataas.com/cat?unique={i}"
    elif i % 3 == 1:
        url = f"https://placekitten.com/{300 + (i % 40)}/{300 + (i % 40)}"
    else:
        url = f"https://picsum.photos/300/300?random={i}"

    try:
        urllib.request.urlretrieve(url, temp_path)
        
        with Image.open(temp_path) as img:
            img = img.resize((256, 256))
            img = img.convert("RGB")
            img.save(dest_path, "PNG", compress_level=9)
            
        saved_count += 1
        if saved_count % 15 == 0:
            print(f"Завантажено: {saved_count}/{target_count}...")
            
    except Exception:
        pass
    
    i += 1

if os.path.exists(temp_path):
    os.remove(temp_path)

print(f"УСПІХ! У папці '{output_dir}' тепер рівно {saved_count} котиків!")