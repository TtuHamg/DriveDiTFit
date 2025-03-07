from tqdm import tqdm
import os
import json
import shutil
import argparse

parser = argparse.ArgumentParser(description="Splite Ithaca365 dataset.")
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--scene_json_path", type=str, required=True)
parser.add_argument("--weather_json_path", type=str, required=True)


def find_weather(weather_json, weather_token):
    for weather_dict in weather_json:
        if weather_dict["token"] == weather_token:
            return weather_dict["description"]
    raise Exception(f"not exist weather token: {weather_token}")


def splite2scenario(dataset_path, save_path, scene_json_path, weather_json_path):
    with open(scene_json_path) as fp:
        scene_json_data = json.load(fp)

    with open(weather_json_path) as fp:
        weather_json_data = json.load(fp)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for scene in scene_json_data:
        date_folder = scene["name"]
        weather_token = scene["weather_token"]

        date_path = os.path.join(dataset_path, date_folder)
        if not os.path.isdir(date_path):
            raise Exception(f"not exist date: {date_folder}")

        weather_desc = find_weather(weather_json_data, weather_token)
        weather_path = os.path.join(save_path, weather_desc)
        if not os.path.isdir(weather_path):
            os.mkdir(weather_path)

        for img_file in os.listdir(date_path):
            shutil.copy(os.path.join(date_path, img_file), weather_path)


args = parser.parse_args()
splite2scenario(
    args.dataset_path, args.save_path, args.scene_json_path, args.weather_json_path
)
