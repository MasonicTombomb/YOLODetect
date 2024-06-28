# detect.py
#
# This module is provided for the detection and segmentation of input images. An input file or list of input files is
# provided and then processed to segment individual crystals.
#
# Author: Thomas Scott
# Institution: Bangor University
# Creation Date: June 27, 2024
# Last Modified: June 27, 2024
# Version: 1.0
# License: MIT License
#
# Copyright (c) 2024 Thomas Scott
#
# Description:
#     This module defines basic functions for detecting and segmenting input images. It provides three main functionalities:
#         - Results file (Always)
#         - Cropped images
#         - Images with bounding boxes
#
# Usage Example:
# ==============
# To run the application, you can execute the following command:
#
#     >>> python3 detect.py -d images/ceria -c True -b True
#
# There are some flags which can be used:
#     -i, --image: Path to the input image. (For Single image)
#     -d, --dir: Directory where images are located. (For Multiple images)
#     -c, --crop: Enables the cropping functionality.
#     -b, --bounding: Enables the bounding functionality.
#
# Classes:
# ========
# (If there are any classes, list them here)
#
# Functions:
# ==========
# segment_images
# crop_images
# draw_bounding_boxes
#
# Usage:
# ======
# (Include a more detailed usage guide here, if necessary.)
#
# Notes:
# ======
# - This module is designed for research purposes and may not cover all edge cases.

#-- Module imports --#
import argparse
import os
import logging
import cv2
import torch
import mimetypes
from pathlib import Path
from tqdm import tqdm

#-- Globals --#
logger = logging.getLogger(__name__)
global ARGS
MIME_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/bmp', 'image/webp']
MODEL = torch.hub.load('ultralytics/yolov5', 'custom', path='data/weights/best.pt')

#-- Setting up Logger --#
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename='detect.log'
                )

#-- Creating parser --#
parser = argparse.ArgumentParser(
        prog='detect',
        description='A simple script to detect crystals within an image.',
        epilog='Example: python detect.py --image <image>'
)
# Defining the arguments which can be passed in
parser.add_argument('-i', '--image', help='Input image.', required=False, type=str)
parser.add_argument('-d', '--dir', help='Input image directory.', required=False, type=str)
#parser.add_argument('-w', '--weights', help='Weights file.', required=False, type=str)
parser.add_argument('-c', '--crop', help='Outputs a the area within bounding boxes.', required=False, type=bool, default=False)
parser.add_argument('-b', '--bounding', help='Outputs an image with the bounding boxes.', required=False, type=bool, default=False)
parser.add_argument('-m', '--metrics', help='Outputs metrics on each image.', required=False, type=bool, default=False)

#-- Model evaluation --#
MODEL.eval()
logger.info('Model loaded.')

class Detect(object):
    def run_detection(self, image, imageName):
        if image is None:
            print(f'Image not found at {imageName}')
            logger.error(f'The Image cannot be found: {imageName}')

        # Running interference
        results = MODEL(image)
        i = 0
        if ARGS.crop or ARGS.metrics:
            predictions = results.xyxy[0].cpu().numpy()
            with tqdm(total=len(predictions), desc=f'Cropping image {imageName}', leave=False) as pbar:
                for pred in predictions:
                    x1, y1, x2, y2, conf, cls = pred
                    data = {
                        'i':     int(i),
                        'x1':    int(x1),
                        'y1':    int(y1),
                        'x2':    int(x2),
                        'y2':    int(y2),
                        'conf':  int(conf),
                        'label': f'{MODEL.names[cls]}: {cls:.2f}'
                    }
                    if ARGS.crop:
                        logger.info(f'Cropping image {imageName}')
                        self.crop_image(image, imageName, data)
                    if ARGS.metrics:
                        logger.info(f'Applying metrics on image {imageName}')
                        self.metrics(image, imageName, data)
                    i += 1
                    pbar.update(1)

        if ARGS.bounding:
            predictions = results.xyxy[0].cpu().numpy()
            with tqdm(total=len(predictions), desc=f'Applying bounding boxes to {imageName}', leave=False) as pbar:
                for pred in predictions:
                    logger.info(f'Applying bounding boxes on image {imageName}')
                    x1, y1, x2, y2, conf, cls = pred
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    bounding_path = Path('boundings')
                    bounding_path.mkdir(parents=True, exist_ok=True)

                    # Construct the file path for the saved image
                    image_file = bounding_path / (Path(imageName).stem + '_boundings.jpg')

                    # Save the image as JPEG
                    cv2.imwrite(str(image_file), image)
                    logger.info(f'Bounding boxes saved on image {imageName}')
                    pbar.update(1)

        resultsPath = Path(f'results/')
        resultsPath.mkdir(parents=True, exist_ok=True)

        resultsFile = f'{resultsPath}/{imageName}.res'
        reses = results.xyxy[0].cpu().numpy()
        with open(resultsFile, 'w') as file:
            # Write variable's string representation to file
            results = str(results)
            file.write(results)
            file.write('\n\n')
            for sublist in reses:
                file.write(f'{str(sublist)} \n')

        logger.info(f'Results saved at {resultsPath}')

    def crop_image(self, image, imageName, data):
        label = data['label']
        i = int(data['i'])
        cropPath = Path(f'crops/{imageName}')
        cropPath.mkdir(parents=True, exist_ok=True)
        cropRegion = image[data['y1']:data['y2'], data['x1']:data['x2']]
        cv2.imwrite(str(cropPath / f'{label}_{i}.jpg'), cropRegion)
        logger.info(f'Cropped image {imageName}, and saved to {cropPath}')

    def metrics(self, image, data):
        pass


def run(imgPath, pathType):
    Det = Detect()
    if not pathType:
        logger.info('Path type not specified, defaulting to single image')
        img = cv2.imread(imgPath)
        imageName = os.path.basename(imgPath)
        Det.run_detection(img, imageName)

    else:
        logger.info('Path type specified, defaulting to multi-image')
        files = [f for f in imgPath.iterdir() if f.is_file() and mimetypes.guess_type(f)[0] in MIME_TYPES]
        with tqdm(total=len(files), desc='Processing images') as pbar:
            for file in files:
                imageName = os.path.basename(file)
                img = cv2.imread(str(file))
                Det.run_detection(img, imageName)
                pbar.update(1)


if __name__ == '__main__':
    # Getting arguments and handling
    ARGS = parser.parse_args()
    if ARGS.image:
        imagePath = Path(ARGS.image)
        pathType = False
    elif ARGS.dir:
        imagePath = Path(ARGS.dir)
        pathType = True
    else:
        raise ValueError('Please provide an input image path or directory path.')
        logger.error('Please provide an input image path or directory path.')
    if os.path.isdir(imagePath):
        run(imagePath, pathType)
        logger.info('Detected images in directory')
    else:
        raise ValueError('Please provide a valid path.')
        logger.error('Please provide a valid path.')
