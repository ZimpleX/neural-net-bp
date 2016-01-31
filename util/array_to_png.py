"""
This script is for converting the npz image array to the png image.
Visualizing the dataset may help debugging.
"""

from PIL import Image
import numpy as np


def array_to_png(path_in, path_out, tag, height, width, num_images=100):
    raw = np.load(path_in)[tag]
    entry = raw.shape[0]
    raw = raw.reshape(entry, height, width)
    stride = entry // num_images
    raw = raw[0::stride, ...]
    rmin = np.min(raw)
    rmax = np.max(raw)
    raw = (255/(rmax-rmin)) * (raw - rmin)
    i = 0
    for img in raw:
        Image.fromarray(np.uint8(img)).save(path_out.format(i))
        i += 1



if __name__ == "__main__":
    array_to_png('train_data/usps.npz', 'img.ignore.visual/usps/{}.png', 
        'validation', 16, 16)
