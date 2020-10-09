import numpy as np
import cv2
from tkinter import *
from tkinter import filedialog
import os
import json 

root = Tk()
root.directoryname = filedialog.askdirectory()
count = 1

def read_config():
    with open('cfg.json') as json_file: 
        data = json.load(json_file) 
    return data

class Augmentations:
    def translation(self, image, t1, t2):
        height, width = image.shape[:2] 
        transaltion_height, transaltion_width = height * t1, width * t2
        T = np.float32([[1, 0, transaltion_width], [0, 1, transaltion_height]])
        img_translation = cv2.warpAffine(image, T, (width, height))
        return img_translation

    def scale(self, image, a1, a2):
        height, width = image.shape[:2] 
        scale_width, scale_height = int(width * a1), int(height * a2)
        img_scaled = cv2.resize(image, (scale_width, scale_height), interpolation = cv2.INTER_AREA)
        return img_scaled

    def shear(self, image, a, b):
        height, width = image.shape[:2] 
        T = np.float32([[1, a, 0], [b, 1, 0]])
        T[0,2] = -T[0,1] * width/2
        T[1,2] = -T[1,0] * height/2
        img_shear = cv2.warpAffine(image, T, (width, height))
        return img_shear

    def rotate(self, image, angle):
        height, width = image.shape[:2] 
        T = cv2.getRotationMatrix2D(((width-1)/2.0,(height-1)/2.0),angle,1)
        img_rotate = cv2.warpAffine(image, T, (width, height))
        return img_rotate

    def flip(self, image, mode):
        height, width = image.shape[:2] 
        img_flip = cv2.flip(image, mode)
        return img_flip

Augments = Augmentations()

def call_augmentation(o, name, image, args):
    kwargs = args
    return getattr(o, name)(image, **kwargs)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append({"path" : filename, "data": img})
    return images

images = load_images_from_folder(root.directoryname)

def write_images_to_folder(folder, images):
    global count
    new_folder = folder + "_aug"
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    config = read_config()['augmentations']

    for aug_cfg in config:
        if aug_cfg['run']:
            aug_name = aug_cfg['name']
            aug_args = aug_cfg['args']
            for image in images:
                new_name = image['path'].replace('.jpg',"_" + aug_name + "_" + str(count) + ".jpg")
                cv2.imwrite(os.path.join(new_folder,new_name), call_augmentation(Augments, aug_name, image['data'], aug_args))
                count += 1

write_images_to_folder(root.directoryname, images)