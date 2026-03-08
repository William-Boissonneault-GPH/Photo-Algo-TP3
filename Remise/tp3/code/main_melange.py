from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
import skimage
from imageio import imwrite


imageMelange_1 = {
    "input_im1_path": 'sourceImages\melange\pommange\\apple.jpeg',
    "input_im2_path": 'sourceImages\melange\pommange\orange.jpeg',
    "input_m_path": 'sourceImages\melange\pommange\mask.jpg',

    "nameOfResult": 'pommange',
    "startStack" : 50,
    "amountStack" : 6,
    "output_dir" : 'resultImages\\melange',
}


###https://www.buzzfeed.com/mjs538/boopgate-explainer
###https://en.wikipedia.org/wiki/The_Creation_of_Adam
imageMelange_2 = {
    "input_im1_path": 'sourceImages\melange\irigulier\\1.png',
    "input_im2_path": 'sourceImages\melange\irigulier\\2.png',
    "input_m_path": 'sourceImages\melange\irigulier\mask.png',

    "nameOfResult": 'I_didnt_touch_adam',
    "startStack" : 100,
    "amountStack" : 7,
    "output_dir" : 'resultImages\\melange',
}

imageMelange_3 = {
    "input_im1_path": 'sourceImages\melange\proporePhoto\\1.png',
    "input_im2_path": 'sourceImages\melange\proporePhoto\\2.png',
    "input_m_path": 'sourceImages\melange\proporePhoto\mask.png',

    "nameOfResult": 'polarization',
    "startStack" : 200,
    "amountStack" : 7,
    "output_dir" : 'resultImages\\melange',
}
###Notice laplacienne mixing create blur compared to simple composition



imageToMelange = imageMelange_3



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

def save_png_auto(img, path, normalize=False):
    """
    Save image to PNG with automatic dtype + range handling.

    Parameters
    ----------
    img : array-like
        Input image (grayscale or RGB)
    path : str
        Output file path
    normalize : bool
        If True, rescale min→0 and max→255 (good for visualization)
        If False, clip to [0,1] then scale (good for physically meaningful images)
    """

    img = np.asarray(img).astype(np.float32)

    # --- Remove NaN / Inf ---
    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    # --- If already uint8 ---
    if img.dtype == np.uint8:
        imwrite(path, img)
        return

    mn, mx = float(img.min()), float(img.max())

    # --- Detect range ---
    if normalize:
        # Stretch contrast to full [0,1]
        if mx > mn:
            img = (img - mn) / (mx - mn)
        else:
            img = np.zeros_like(img)

    else:
        # Physical mode: clip to valid [0,1]
        if mx > 1.0 or mn < 0.0:
            img = np.clip(img, 0.0, 1.0)

        # If it looks like already in 0–255, convert down
        elif mx > 1.0:
            img = img / 255.0

    # --- Convert to uint8 ---
    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)

    # --- Debug warnings ---
    if img_u8.max() == 0:
        print("[WARNING] saved image is all black")
    if img_u8.min() == 255:
        print("[WARNING] saved image is all white")

    imwrite(path, img_u8)





def pile_laplaciennce(image_path, start, amount, show=False):

    image = imread(image_path)
    ###remove alpha channel
    image = image[..., :3]

    gaussian_stack = []
    fc_list = []  # cycles/image at each Gaussian level

    for i in range(amount):
        print(i)
        sigma = sigma_from_cycles_per_image(start / (2**i),image.shape[0],image.shape[1])
        fc_list.append(start / (2**i))
        gaussian_stack.append(skimage.filters.gaussian(image, sigma, channel_axis=-1))
    
    laplace_stack = []
    band_list = []
    
    for i in range(1,amount):
        laplace_stack.append(gaussian_stack[i-1]-gaussian_stack[i])

        high_fc = fc_list[i-1]
        low_fc  = fc_list[i]
        band_list.append((low_fc, high_fc))
    

    return laplace_stack, gaussian_stack[-1]

def pile_gaussienne(image_path, start, amount, show=False):

    image = imread(image_path)
    ###remove alpha channel
    image = image[..., :3]

    gaussian_stack = []
    fc_list = []  # cycles/image at each Gaussian level

    for i in range(amount):
        sigma = sigma_from_cycles_per_image(start / (2**i),image.shape[0],image.shape[1])
        fc_list.append(start / (2**i))
        gaussian_stack.append(skimage.filters.gaussian(image, sigma, channel_axis=None))

    return gaussian_stack




def normalize_for_display(img):
    """Normalise une image pour affichage [0,1] même si valeurs négatives."""
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)

def show_ingredients(laplace_1, gaussienne_m, laplace_2, residuals):

    n_levels = len(laplace_1)+1

    plt.figure(figsize=(3*n_levels, 3*3))  # 3 lignes

    # ---------- ROW 1 : Laplace 1 ----------
    for i, img in enumerate(laplace_1):
        plt.subplot(3, n_levels, i+1)
        plt.imshow(normalize_for_display(img))
        plt.title(f"L1-{i}")
        plt.axis('off')
    plt.subplot(3, n_levels, n_levels)
    plt.imshow(normalize_for_display(residuals[0]))
    plt.title(f"Residue 1-{i}")
    plt.axis('off')
    

    # ---------- ROW 2 : Gaussienne mask ----------
    for i, img in enumerate(gaussienne_m):
        plt.subplot(3, n_levels, n_levels + i + 1)
        plt.imshow(normalize_for_display(img))
        plt.title(f"Gm-{i}")
        plt.axis('off')
    

    # ---------- ROW 3 : Laplace 2 ----------
    for i, img in enumerate(laplace_2):
        plt.subplot(3, n_levels, 2*n_levels + i + 1)
        plt.imshow(normalize_for_display(img))
        plt.title(f"L2-{i}")
        plt.axis('off')
    plt.subplot(3, n_levels, 3*n_levels)
    plt.imshow(normalize_for_display(residuals[1]))
    plt.title(f"Residue 2-{i}")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{imageToMelange['output_dir']}\\{imageToMelange['nameOfResult']}_laplaceStack.png")
    plt.show()




laplace_stack_1, residual_1 = pile_laplaciennce(imageToMelange['input_im1_path'],imageToMelange['startStack'],imageToMelange['amountStack'])
laplace_stack_2, residual_2 = pile_laplaciennce(imageToMelange['input_im2_path'],imageToMelange['startStack'],imageToMelange['amountStack'])
gaussien_stack_m = pile_gaussienne(imageToMelange['input_m_path'],imageToMelange['startStack'],imageToMelange['amountStack'])

#laplace_stack_1 = pile_laplaciennce(imageToMelange['input_im1_path'],100,7)

show_ingredients(laplace_stack_1,gaussien_stack_m,laplace_stack_2, [residual_1,residual_2])

###Compute laplace merged substack
melange_pile = []

for i in range(len(laplace_stack_1)):
    #I1*masque + I2*(1-masque)
    t_im1 = laplace_stack_1[i]
    t_im2 = laplace_stack_2[i]
    t_m = gaussien_stack_m[i]

    tranche = t_im1 * t_m + t_im2 * (1-t_m)
    melange_pile.append(tranche)
#residual
t_im1 = residual_1
t_im2 = residual_2
t_m = gaussien_stack_m[-1]
tranche = t_im1 * t_m + t_im2 * (1-t_m)
melange_pile.append(tranche)


plt.figure(figsize=(3*len(melange_pile),4))

for i, img in enumerate(melange_pile):
    plt.subplot(1, len(melange_pile), i+1)
    plt.imshow(normalize_for_display(img), cmap='gray')
    plt.axis('off')
plt.tight_layout()
#plt.savefig(f'{output_dir}_gaussienne.png')
plt.savefig(f"{imageToMelange['output_dir']}\\{imageToMelange['nameOfResult']}_combination.png")
plt.show()

out = np.zeros_like(melange_pile[0], dtype=np.float32)
for tranche in melange_pile:
    out += tranche

save_png_auto(out,f"{imageToMelange['output_dir']}\\{imageToMelange['nameOfResult']}_final.png")
plt.imshow(out)
plt.show()