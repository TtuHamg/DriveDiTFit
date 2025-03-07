from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description="Reszie Ithaca365 dataset.")
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)


def convert_resolution_256(datasets_folder, img_folder, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    transform = transforms.Resize([256, 256])
    for date_folder in tqdm(os.listdir(datasets_folder)):
        save_folder = os.path.join(save_path, date_folder)
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        date_path = os.path.join(datasets_folder, date_folder)
        for cam_folder in os.listdir(date_path):
            if cam_folder != img_folder:
                continue

            cam_path = os.path.join(date_path, cam_folder)
            for img_file in os.listdir(cam_path):
                img = Image.open(os.path.join(cam_path, img_file))
                img_256 = transform(img)
                img_256.save(os.path.join(save_folder, img_file))


args = parser.parse_args()
convert_resolution_256(args.dataset_path, "cam0", args.save_path)
