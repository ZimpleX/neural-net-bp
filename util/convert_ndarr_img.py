"""
This script is for converting the npz image array to the png image.
Visualizing the dataset may help debugging.
"""

from PIL import Image
import numpy as np
import copy
import pdb


def array_to_img(path_in, path_out, tag, channel, height, width, num_images=100, scale_individual=True):
    raw = np.load(path_in)[tag]
    entry = raw.shape[0]
    if channel == 1:
        raw = raw.reshape(entry, height, width)
    else:
        raw = raw.reshape(entry, channel, height, width)
    stride = entry // num_images
    raw = raw[0::stride, ...]
    if scale_individual:
        for i in range(len(raw)):
            rmin = np.min(raw[i])
            rmax = np.max(raw[i])
            raw[i] = (255/(rmax-rmin)) * (raw[i] - rmin)
    else:
        rmin = np.min(raw)
        rmax = np.max(raw)
        raw = (255/(rmax-rmin)) * (raw - rmin)
    i = 0
    for img in raw:
        Image.fromarray(np.uint8(img)).save(path_out.format(i))
        i += 1

def img_to_array(path_img):
    arr_img = np.asarray(Image.open(path_img))
    # scale to -1 ~ 1
    arr_img = copy.deepcopy(arr_img)
    arr_img = arr_img/127.5 - 1.
    # reshape
    shape = arr_img.shape
    if len(shape) == 2:
        return arr_img.reshape(1,1,*shape)
    else:
        return arr_img.transpose((2,0,1))[np.newaxis, ...]
    


if __name__ == "__main__":
    array_to_img('train_data/3T3_2500.npz', 'img.ignore.visual/bloodcell/3T3_scale/{}.png', 
        'data', 1 , 256, 256, scale_individual=True)
