#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_dataset.py
This script generates a dataset for training a model. It uses a folder of digits to create a COCO-style dataset by
drawing the digits on random backgrounds and saving the images and annotations in the required format.
"""

import cv2
import numpy as np
from tqdm import tqdm

import os
import random

def get_background_from_directory(directory="./datasets/backgrounds"):
    """
    Selects a random background image from the specified directory and returns a random 512x512 crop of it.
    """
    max_attempt = 100
    for _ in range(max_attempt):
        background_files = [f for f in os.listdir(directory) if f.endswith((".jpg", ".png"))]
        if not background_files:
            raise ValueError("No background images found in the specified directory.")
        background_file = random.choice(background_files)
        background_path = os.path.join(directory, background_file)
        background = cv2.imread(background_path)
        if background is None:
            raise ValueError("Failed to read background image.")

        h, w, _ = background.shape

        # Ensure crop doesn't containt OSD dates
        x = random.randint(500, w - 512)
        y = random.randint(0, h - 512 - 200)
        candidate = background[y:y + 512, x:x + 512]
        if np.average(candidate) < 100:
            continue
        return candidate
    else:
        raise Exception("No valid background found!")

def create_coco_dataset(digits_folder, output_folder, num_train=1000, num_val=200):
    # Set up COCO-style directory structure
    images_train_dir = os.path.join(output_folder, "COCO", "images", "train2017")
    images_val_dir = os.path.join(output_folder, "COCO", "images", "val2017")
    labels_train_dir = os.path.join(output_folder, "COCO", "labels", "train2017")
    labels_val_dir = os.path.join(output_folder, "COCO", "labels", "val2017")
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    
    total_black_digits = 0
    total_white_digits = 0

    def generate_split(num_images, images_dir, labels_dir):
        nonlocal total_black_digits, total_white_digits

        digit_files = os.listdir(digits_folder)
        noise = np.zeros((512, 512, 3), np.uint8)

        for i in tqdm(range(num_images)):
            background = get_background_from_directory()
            num_digits = random.randint(0, 12)
            used_boxes = []
            label_lines = []
            for _ in range(num_digits):
                digit_file = random.choice(digit_files)
                digit_path = os.path.join(digits_folder, digit_file)
                digit_image = cv2.imread(digit_path, cv2.IMREAD_UNCHANGED)
                if digit_image is None:
                    continue
                scale_factor = random.uniform(0.1, 1.5)
                digit_image = cv2.resize(
                    digit_image,
                    None,
                    fx=scale_factor,
                    fy=scale_factor,
                    interpolation=cv2.INTER_LINEAR,
                )
                digit_color = random.choice(["black", "white"])
                max_attempts = 20
                for attempt in range(max_attempts):
                    x_offset = random.randint(
                        0, background.shape[1] - digit_image.shape[1]
                    )
                    y_offset = random.randint(
                        0, background.shape[0] - digit_image.shape[0]
                    )
                    bbox = [
                        x_offset,
                        y_offset,
                        digit_image.shape[1],
                        digit_image.shape[0],
                    ]
                    overlap = False
                    for ub in used_boxes:
                        x1, y1, w1, h1 = ub
                        x2, y2, w2, h2 = bbox
                        if not (
                            x2 + w2 < x1 or x2 > x1 + w1 or y2 + h2 < y1 or y2 > y1 + h1
                        ):
                            overlap = True
                            break
                    if not overlap:
                        used_boxes.append(bbox)
                        break
                else:
                    continue
                if digit_image.ndim == 2 or (
                    digit_image.ndim == 3 and digit_image.shape[2] == 1
                ):
                    mask = digit_image == 0
                    if digit_color == "black":
                        for c in range(3):
                            background[
                                y_offset : y_offset + digit_image.shape[0],
                                x_offset : x_offset + digit_image.shape[1],
                                c,
                            ][mask] = 0
                    else:
                        for c in range(3):
                            background[
                                y_offset : y_offset + digit_image.shape[0],
                                x_offset : x_offset + digit_image.shape[1],
                                c,
                            ][mask] = 255
                elif digit_image.ndim == 3 and digit_image.shape[2] == 4:
                    alpha = digit_image[:, :, 3] / 255.0
                    for c in range(3):
                        background[
                            y_offset : y_offset + digit_image.shape[0],
                            x_offset : x_offset + digit_image.shape[1],
                            c,
                        ] = (
                            background[
                                y_offset : y_offset + digit_image.shape[0],
                                x_offset : x_offset + digit_image.shape[1],
                                c,
                            ]
                            * (1 - alpha)
                            + digit_image[:, :, c] * alpha
                        ).astype(
                            np.uint8
                        )
                else:
                    if digit_color == "black":
                        mask = np.any(digit_image != 0, axis=2)
                        for c in range(3):
                            background[
                                y_offset : y_offset + digit_image.shape[0],
                                x_offset : x_offset + digit_image.shape[1],
                                c,
                            ][mask] = digit_image[:, :, c][mask]
                    else:
                        mask = np.any(digit_image == 255, axis=2)
                        for c in range(3):
                            background[
                                y_offset : y_offset + digit_image.shape[0],
                                x_offset : x_offset + digit_image.shape[1],
                                c,
                            ][mask] = 255
                # YOLO label: <class_id> <x_center> <y_center> <width> <height> (normalized)
                class_id = int(digit_file[-5])
                x = x_offset
                y = y_offset
                w = digit_image.shape[1]
                h = digit_image.shape[0]
                img_w = background.shape[1]
                img_h = background.shape[0]
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Check if image is on a fully black or white background making it unsuitable for training
                # Using every digit as a template and match against the background
                patch = background[y:y + h, x:x + w].copy()
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                results = []
                for digit in digit_files:
                    digit_path = os.path.join(digits_folder, digit)
                    digit_img = cv2.imread(digit_path, cv2.IMREAD_UNCHANGED)

                    digit_img = cv2.resize(
                        digit_img,
                        (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    img = cv2.matchTemplate(patch, digit_img, cv2.TM_CCOEFF_NORMED)
                    _, result, _, _ = cv2.minMaxLoc(img)

                    results.append(abs(result))

                max_result_index = np.argmax(results)
                if results[max_result_index] > 0.5 and int(digit_files[max_result_index][-5]) == class_id:
                    label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                    if digit_color == 'black':
                        total_black_digits += 1
                    else:
                        total_white_digits += 1

            image_filename = f"image_{i:020d}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            label_filename = f"image_{i:020d}.txt"
            label_path = os.path.join(labels_dir, label_filename)

            # Blur and Add noise
            blurred_img = cv2.GaussianBlur(background, (5, 5), 0)
            cv2.randn(noise, 0, 256)
            final_img = cv2.add(blurred_img, noise)

            cv2.imwrite(image_path, final_img)
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines) + "\n")

    generate_split(num_train, images_train_dir, labels_train_dir)
    generate_split(num_val, images_val_dir, labels_val_dir)
    print(f"Total black digits: {total_black_digits}\nTotal white digits: {total_white_digits}")


if __name__ == "__main__":
    digits_folder = "./digits"  # Replace with the path to your digits folder
    output_folder = "/data/datasets"  # Replace with your desired output folder
    create_coco_dataset(digits_folder, output_folder, num_train=50000, num_val=10000)
    # create_coco_dataset(digits_folder, output_folder, num_train=50, num_val=10)
    print(f"Training and validation datasets generated in {output_folder}/COCO")
