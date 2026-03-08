import matplotlib.pyplot as plt
import numpy as np


def crop_image(img, img2):
    print('Select two points that define the area of the image you '
          'want to crop')
    plt.imshow(img, cmap='gray')
    x, y = tuple(zip(*plt.ginput(2)))
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    img1 = img[min(y):max(y), min(x):max(x)]
    img2 = img2[min(y):max(y), min(x):max(x)]


    return img1, img2
