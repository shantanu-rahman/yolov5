#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The fetchbg.py script is designed to fetch and prepare background images for a dataset generation process.
It reads video files from a specified directory, extracts random frames from these videos, and saves them as images in a designated output folder.
"""

import os
import argparse
import av
import random

def fetch_backgrounds(video_folder, output_folder, num_frames=100):
    os.makedirs(output_folder, exist_ok=True)
    video_files = []
    for root, _, files in os.walk(video_folder):
        for f in files:
            if f.endswith((".mp4", ".avi", ".mov")):
                video_files.append(os.path.join(root, f))
    

    frames_extracted = 0
    while frames_extracted < num_frames:
        video_path = random.choice(video_files)
        video_file = os.path.basename(video_path)
        container = av.open(video_path)
        packets = list(filter(lambda p: p.dts is not None and p.is_keyframe, container.demux(video=0)))
        packet = random.choice(packets)
        try:
            frame = packet.decode_one()
            if frame is not None:
                image = frame.to_image()
                image_name = f"{os.path.splitext(video_file)[0]}_{packet.dts}.jpg"
                image_path = os.path.join(output_folder, image_name)
                image.save(image_path)
                print(f"Saved {image_path}")
                frames_extracted += 1
                if frames_extracted >= num_frames:
                    break
        except av.error.InvalidDataError:
            print(f"Invalid data in video {video_file}, skipping frame extraction.")
            continue

def main():
    parser = argparse.ArgumentParser(description="Fetch background images from video files.")
    parser.add_argument("video_folder", type=str, help="Path to the folder containing video files.")
    parser.add_argument("output_folder", type=str, help="Path to the folder where extracted images will be saved.")
    parser.add_argument("--num_frames", type=int, default=32, help="Number of frames to extract from each video.")
    
    args = parser.parse_args()
    
    fetch_backgrounds(args.video_folder, args.output_folder, args.num_frames)

if __name__ == "__main__":
    main()