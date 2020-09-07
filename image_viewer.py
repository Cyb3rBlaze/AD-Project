import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]



def make_small(img, width, height):
    new_image = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    return new_image


def view_sample_image(filename):
    dataset = pydicom.dcmread(filename)

    plt.imsave("sample.jpg", crop_center(dataset.pixel_array, 128, 128))

view_sample_image("sample.dcm")