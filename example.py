from cspace import CSpace
# from cspace import CSpaceCached as CSpace

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
    shape = imgs[0].shape
    # imgs = [i.tolist() for i in imgs]  # test if this is faster?

    # Create C-Space filter
    c = CSpace(shape, 512, 0.20, 150, 0.1)

    imgs_out = []
    for i in range(len(imgs)):
        tstart = time.perf_counter()
        img_expanded = c.filter(imgs[i])
        tend = time.perf_counter()
        print(f'Time: {tend - tstart}')
        imgs_out.append(c.filter(imgs[i]))

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
        plt.imshow(imgs[i], vmin=0, vmax=100)
        plt.subplot(len(imgs), 2, 2 + 2 * i)
        plt.imshow(imgs_out[i], vmin=0, vmax=100)
    plt.show()


if __name__ == '__main__':
    main()
