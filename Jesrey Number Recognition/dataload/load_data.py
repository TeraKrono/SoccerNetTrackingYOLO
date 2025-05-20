import os
import urllib.request
import zipfile

# Папка, куда всё сохраняется
base_path = "../pv2_task(j-n-r)/SoccerNet"
os.makedirs(base_path, exist_ok=True)

# Файлы для скачивания
downloads = [
    {
        "url": "https://download.soccer-net.org/jersey/train/jersey_train.zip",
        "out": os.path.join(base_path, "jersey_train.zip"),
        "extract_to": os.path.join(base_path, "train")
    },
    {
        "url": "https://download.soccer-net.org/jersey/train/labels_train.zip",
        "out": os.path.join(base_path, "labels_train.zip"),
        "extract_to": os.path.join(base_path, "train")
    },
    {
        "url": "https://download.soccer-net.org/jersey/test/jersey_test.zip",
        "out": os.path.join(base_path, "jersey_test.zip"),
        "extract_to": os.path.join(base_path, "test")
    },
    {
        "url": "https://download.soccer-net.org/jersey/test/labels_test.zip",
        "out": os.path.join(base_path, "labels_test.zip"),
        "extract_to": os.path.join(base_path, "test")
    },
]

for item in downloads:
    print(f"📥 Downloading: {item['url']}")
    urllib.request.urlretrieve(item["url"], item["out"])
    print(f"✅ Extracting to: {item['extract_to']}")
    os.makedirs(item["extract_to"], exist_ok=True)
    with zipfile.ZipFile(item["out"], 'r') as zip_ref:
        zip_ref.extractall(item["extract_to"])
    os.remove(item["out"])

print("✅ All downloads and extractions complete.")
