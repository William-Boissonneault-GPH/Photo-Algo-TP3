import glob
import os
import numpy as np
import skimage
from imageio import imread
from tp1_startcode.tp1_io import quantize_to_8bit, normal_image
from tp1_startcode.tp1_rapport import (
    html_document,
    section,
    subsection,
    figure,
    table,
    algorithm_box,
    save_report,
    create_demosaic_comparison_figure,
    create_difference_figure,
    find_edge_region,
    create_demosaic_zoom_figure,
)



import matplotlib.pyplot as plt

"""ce code principal prend toutes les images du input dir et applique du sharppening, fait une figure de comparaison du niveau de sharpening utilisé et recrache des images du niveau moyen dans output direct

Crée aussi le template d'une section de rapport"""


####################SETUP UP#######################
images_a_accentuer_input_dir = 'sourceImages\\accentuation'
images_a_accentuer_output_dir = 'resultImages\\accentuation'
########Voir des paramètres variables, ou seulement le rendu final#############
create_params_finding_figure = False


###ROIS selected for comparison figure
def _norm_path(p):
    return os.path.normcase(os.path.normpath(p))
crop_roi_by_path = {
    'sourceimages\\accentuation\\aegep_2025__20.jpg': (1938, 2301, 1026, 738),
    'sourceimages\\accentuation\\dsc_9683_final.jpg': (2783, 1387, 1081, 832),
    'sourceimages\\accentuation\\dsc_0254_final_tuned.png': (2444, 3020, 1426, 2118),
}





extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]

#va chercher tout les chemins d'image
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(images_a_accentuer_input_dir, ext)))


def applySharpening(image, spatial_sigma, alpha):
    print(image.shape)
    float_img = image.astype(np.float32) / 255.0
    gaussian_filtered = skimage.filters.gaussian(float_img, sigma=spatial_sigma, channel_axis=-1)
    sharpenned_image = float_img + alpha * (float_img - gaussian_filtered)
    return quantize_to_8bit(sharpenned_image)

######passe à travers chaque image############

i=0
for img_path in image_paths:
    image_init = imread(img_path)
    i+=1
    ##############Figure de choix de paramètres#######################
    if create_params_finding_figure:
        images={}
                        
        images[f"initial"] = image_init
        sigmas_spatial_to_apply = [2, 4, 8, 16, 32, 64]
        for sigma_spatial_to_apply in sigmas_spatial_to_apply:
            images[f"sharpenning spatial sigma = {sigma_spatial_to_apply}"] = applySharpening(image_init, sigma_spatial_to_apply, 0.5)
            print("good day sir")

        # Figure de zoom
        edge_pos = find_edge_region(image_init)
        center_pos = (2*image_init.shape[0] // 3, image_init.shape[1] // 2)
        create_demosaic_zoom_figure(
            images,
            edge_pos,
            center_pos,
            os.path.join(images_a_accentuer_output_dir, f"image{i}_sharppening_var_sigma_spatial.png"),
            normal_image,
            title=f"Zoom - image {i} - ; alpha = 0.5",
        )

    if create_params_finding_figure:
        images={}
                        
        images[f"initial"] = image_init
        alphas_to_apply = [0.1, 0.2, 0.4, 0.8, 1]
        for alpha_to_apply in alphas_to_apply:
            images[f"sharpenning alpha = {alpha_to_apply}"] = applySharpening(image_init, 16, alpha_to_apply)
            print("good day sir")

        # Figure de zoom
        edge_pos = find_edge_region(image_init)
        center_pos = (2*image_init.shape[0] // 3, image_init.shape[1] // 2)
        create_demosaic_zoom_figure(
            images,
            edge_pos,
            center_pos,
            os.path.join(images_a_accentuer_output_dir, f"image{i}_sharppening_var_alpha.png"),
            normal_image,
            title=f"Zoom - image {i} - ; sigma spatial = 16",
        )

    else:
    #############Application des paramètres choisis##############
        sigma_spatial_choisi = 16
        alpha_choisi = 0.5

        img = image_init
        roi = crop_roi_by_path.get(_norm_path(img_path), None)
        if roi is not None:
            x, y, w, h = roi
            img = img[y:y+h, x:x+w]

        # Figure de comparaison
        print("  Création des figures de comparaison...")
        images = {"Initial": img}
        images[f"Sharpened (sigma spatial : {sigma_spatial_choisi}), alpha : {alpha_choisi}"] = applySharpening(img, sigma_spatial_choisi, alpha_choisi)

        create_demosaic_comparison_figure(
            images,
            os.path.join(images_a_accentuer_output_dir, f"c_image{i}_comparison.png"),
            normal_image,
            title=f"Avant/Après Sharpening - image{i}",
        )


