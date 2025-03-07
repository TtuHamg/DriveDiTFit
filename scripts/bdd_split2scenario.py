import json
import os
import shutil
from PIL import Image
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Splite BDD100K dataset.")
parser.add_argument("--train_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--label_json_path", type=str, required=True)

args = parser.parse_args()

f = open(args.label_json_path, "r")
content = f.read()
label_json = json.loads(content)


dir = args.train_path
root = args.save_path

ignore_weather = ["undefined", "foggy"]

for index, item in tqdm(enumerate(label_json)):
    name = item["name"]
    weather = item["attributes"]["weather"]
    if weather in ignore_weather:
        continue
    if item["attributes"]["timeofday"] == "night":
        weather = item["attributes"]["timeofday"]
    source_path = os.path.join(dir, name)
    target_path = os.path.join(root, weather)
    os.makedirs(target_path, exist_ok=True)

    img = Image.open(source_path)
    out = img.resize((256, 256), Image.LANCZOS)
    out.save(f"{target_path}/{name}")
