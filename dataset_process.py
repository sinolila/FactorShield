import os
import shutil
import random
from PIL import Image
import torch
import torch.nn as nn
from train_face import *
from concurrent.futures import ThreadPoolExecutor


def copy_images(source_dir, target_dir, percentage=0.8):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    all_files = os.listdir(source_dir)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    num_images_to_copy = int(len(image_files) * percentage)
    images_to_copy = random.sample(image_files, num_images_to_copy)

    for image in images_to_copy:
        source_path = os.path.join(source_dir, image)
        target_path = os.path.join(target_dir, image)
        shutil.copy(source_path, target_path)

    print(f"Copied {len(images_to_copy)} images to {target_dir}")



def process_image(input_path, output_path, size):
    try:
        with Image.open(input_path) as img:
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            img_resized.save(output_path)
        print(f"Resized and saved: {output_path}")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")


def resize_and_split(input_dir, train_dir, val_dir, size=(256, 256), num_workers=4, train_ratio=0.7):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    random.shuffle(files)
    train_count = int(len(files) * train_ratio)
    train_files = files[:train_count]
    val_files = files[train_count:]
    def worker(filename, output_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            print(f"Skipping already processed file: {filename}")
            return
        process_image(input_path, output_path, size)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        train_futures = [executor.submit(worker, f, train_dir) for f in train_files]
        val_futures = [executor.submit(worker, f, val_dir) for f in val_files]

        for future in train_futures + val_futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")


def resize_images(input_dir, size=(256, 256), num_workers=4):
    for split in ['tra/train', 'val/valid']:
        split_dir = os.path.join(input_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f'{split} directory does not exist in {input_dir}')

        output_dir = f'{split_dir}_resized'
        os.makedirs(output_dir, exist_ok=True)

        files = [f for f in os.listdir(split_dir) if f.endswith(".png")]

        def worker(filename):
            input_path = os.path.join(split_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                print(f"Skipping already processed file: {filename}")
                return

            process_image(input_path, output_path, size)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(worker, files)
        print(f'Resized images saved to {output_dir}')
