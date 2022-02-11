from cspace import CSpace
# from cspace import CSpaceCached as CSpace
from cspace import FastArray

import time
import numpy as np
import PIL.Image

import matplotlib.pyplot as plt


def load_image(path):
    img = np.array(PIL.Image.open(path))
    assert(len(img.shape) == 2)
    return img


def main():
    # Load test images
    imgs = [
        load_image('26.png'),
        load_image('114.png'),
        load_image('120.png')
    ]

    downsample_factor = 3
    for i, img in enumerate(imgs):
        imgs[i] = (img[0:-1:downsample_factor, 0:-1:downsample_factor] / downsample_factor).astype(int)

    shape = imgs[0].shape

    # for i, img in enumerate(imgs):
    #     a = FastArray(shape)
    #     for x in range(shape[1]):
    #         for y in range(shape[0]):
    #             a[y, x] = img[y, x]
    #     imgs[i] = a

    # Create C-Space filter
    disparities = 128
    baseline = 0.50
    focal_length = np.sqrt(shape[0]**2 + shape[1]**2)
    radius = 0.2
    c = CSpace(shape, disparities, baseline, focal_length, radius)

    imgs_out = []
    for i in range(len(imgs)):
        tstart = time.perf_counter()
        img_expanded = c.filter(imgs[i])
        tend = time.perf_counter()
        print(f'Time: {tend - tstart}')
        imgs_out.append(img_expanded)

    # Comparison: test numpy read/write speed
    img_test = imgs[0].copy()
    tstart = time.perf_counter()
    for x in range(shape[1]):
        for y in range(shape[0]):
            img_test[y, x] = 1.1 * img_test[y, x]
    tend = time.perf_counter()
    print(f'Numpy access time: {tend - tstart}')

    # Show result
    plt.figure()
    for i in range(len(imgs)):
        plt.subplot(len(imgs), 2, 1 + 2 * i)
        plt.imshow(imgs[i], vmin=0, vmax=100 / downsample_factor)
        plt.subplot(len(imgs), 2, 2 + 2 * i)
        plt.imshow(imgs_out[i], vmin=0, vmax=100 / downsample_factor)
    plt.show()


if __name__ == '__main__':
    main()
