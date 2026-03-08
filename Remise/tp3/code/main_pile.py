#from main_hybride import sigma_from_cycles_per_image

from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
import skimage

input_img = 'sourceImages\pile\\albert_monroe_finalImg.png'
output_dir = 'resultImages\pile\\albertMonroe'


def sigma_from_cycles_per_image(cycles_per_image: float, H: int, W: int) -> float:
    """
    Vient de chat-GPT
    Convert cutoff in cycles/image to spatial Gaussian sigma (pixels),
    using -6 dB (|H|=0.5) cutoff definition and isotropic N=min(H,W).
    """
    N = min(H, W)
    fc = cycles_per_image / N  # cycles/pixel
    sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * fc)
    return float(sigma)


def pile_gaussienne_laplaciennce(image_path, start, amount):

    image = imread(input_img, pilmode='L')

    gaussian_stack = []
    fc_list = []  # cycles/image at each Gaussian level

    for i in range(amount):
        
        sigma = sigma_from_cycles_per_image(start / (2**i),image.shape[0],image.shape[1])
        fc_list.append(start / (2**i))
        gaussian_stack.append(skimage.filters.gaussian(image, sigma, channel_axis=None))
    
    plt.figure(figsize=(3*len(gaussian_stack),4))
    for i, img in enumerate(gaussian_stack):
        plt.subplot(1, len(gaussian_stack), i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"G{i}\nfc={fc_list[i]:.2f} cyc/img", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}_gaussienne.png')
    plt.show()

    laplace_stack = []
    band_list = []
    
    for i in range(1,amount):
        laplace_stack.append(gaussian_stack[i-1]-gaussian_stack[i])

        high_fc = fc_list[i-1]
        low_fc  = fc_list[i]
        band_list.append((low_fc, high_fc))


    plt.figure(figsize=(3*len(laplace_stack),4))
    for i, img in enumerate(laplace_stack):
        img_disp = (img - img.min()) / (img.max() - img.min())

        low_fc, high_fc = band_list[i]

        plt.subplot(1, len(laplace_stack), i+1)
        plt.imshow(img_disp, cmap='gray')
        plt.title(f"L{i}\n{high_fc:.2f}–{low_fc:.2f} cyc/img", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}_laplacienne.png')
    plt.show()


    pass


import numpy as np
from skimage.filters import gaussian
from skimage.transform import resize




#################IMPLEMENTATION CHAT GPT#################
def gaussian_pyramid(img_path, n_levels=5, sigma=1.0, downscale=2):
    """
    Construit une pyramide gaussienne:
      G0 = img
      G_{k+1} = downsample( gaussian(G_k, sigma) )

    Params
    ------
    img : ndarray (H,W) ou (H,W,C)
        Image d'entrée (uint8 ou float).
    n_levels : int
        Nombre de niveaux à retourner (incluant le niveau 0).
    sigma : float
        Sigma du flou gaussien appliqué avant chaque downsample.
    downscale : int
        Facteur de réduction (2 = diviser H et W par 2).

    Returns
    -------
    pyr : list of ndarray
        Liste [G0, G1, ..., G_{n_levels-1}] en float32.
        Valeurs dans [0,1] si l'entrée était uint8, sinon conservées (clip si besoin).
    """

    img = imread(img_path)
    if n_levels < 1:
        raise ValueError("n_levels doit être >= 1")

    # --- Convertir en float32 ---
    img_f = img.astype(np.float32)
    if img_f.dtype != np.float32:
        img_f = img_f.astype(np.float32)

    # Si uint8 classique, on normalise vers [0,1]
    if img.dtype == np.uint8:
        img_f /= 255.0

    pyr = [img_f]

    for _ in range(1, n_levels):
        prev = pyr[-1]

        # 1) blur
        # channel_axis = -1 pour RGB, None pour grayscale
        channel_axis = -1 if prev.ndim == 3 else None
        blurred = gaussian(prev, sigma=sigma, channel_axis=channel_axis, preserve_range=True)

        # 2) downsample (anti-aliasing déjà aidé par blur, mais on garde resize propre)
        H, W = blurred.shape[:2]
        newH = max(1, H // downscale)
        newW = max(1, W // downscale)

        if blurred.ndim == 2:
            reduced = resize(
                blurred, (newH, newW),
                order=1, mode="reflect", anti_aliasing=True, preserve_range=True
            ).astype(np.float32)
        else:
            C = blurred.shape[2]
            reduced = resize(
                blurred, (newH, newW, C),
                order=1, mode="reflect", anti_aliasing=True, preserve_range=True
            ).astype(np.float32)

        pyr.append(reduced)

    return pyr


pyr = gaussian_pyramid(input_img,7)
plt.figure(figsize=(15, 4))
for i, level in enumerate(pyr):
    plt.subplot(1, len(pyr), i+1)
    plt.imshow(level, cmap="gray" if level.ndim == 2 else None)
    plt.title(f"G{i}\n{level.shape[1]}x{level.shape[0]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig(f'{output_dir}_pyr_gaussienne.png')
plt.show()

#pile_gaussienne_laplaciennce(input_img, 100, 7)