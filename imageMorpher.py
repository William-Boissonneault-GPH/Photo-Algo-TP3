import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def Load_pts_from_txt_files(txt_file_path):

    points_array = []
    with open(txt_file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) >= 2:
                x = float(parts[0])
                y = float(parts[1])
                points_array.append((x, y))

    return np.array(points_array)


img1_file_name = '16_Saignes_Benjamin'
img2_file_name = '17_Boissonneault_William'

img1_pts = Load_pts_from_txt_files(f'sourceImages\\faceMerge\\{img1_file_name}.txt')
img2_pts = Load_pts_from_txt_files(f'sourceImages\\faceMerge\\{img2_file_name}.txt')

img1 = mpimg.imread(f"sourceImages\\faceMerge\\{img1_file_name}.jpg")
img2 = mpimg.imread(f"sourceImages\\faceMerge\\{img2_file_name}.jpg")

def calculate_mean_tri(img1_pts, img2_pts):
    img1_5_pts = (img1_pts + img2_pts) / 2
    return Delaunay(img1_5_pts)

def morph(img1, img2, img1_pts, img2_pts, tri, warp_frac=0.5, dissolve_frac=0.5, showFig=False, saveFig=False):

    name='Face_merge'

    ###Montrer les points
    if (showFig):
        ###Première figure - montrer les points des deux images
        fig, ax = plt.subplots(1, 2, figsize=(12,6))

        # ---- First image ----
        ax[0].imshow(img1, extent=[0,720,0,720], origin='lower')
        #ax[0].triplot(img1_pts[:,0], img1_pts[:,1], tri.simplices)
        ax[0].scatter(img1_pts[:,0], img1_pts[:,1],
              c=np.arange(len(img1_pts)),
              cmap='tab20',
              s=40)
        ax[0].set_xlim(0,720)
        ax[0].set_ylim(720,0)
        ax[0].set_title("Image 1")

        # ---- Second image ----
        ax[1].imshow(img2, extent=[0,720,0,720], origin='lower')
        #ax[1].triplot(img2_pts[:,0], img2_pts[:,1], tri.simplices)
        ax[1].scatter(img2_pts[:,0], img2_pts[:,1],
              c=np.arange(len(img2_pts)),
              cmap='tab20',
              s=40)
        ax[1].set_xlim(0,720)
        ax[1].set_ylim(720,0)
        ax[1].set_title("Image 2")

        plt.tight_layout()
        if (saveFig):
            plt.savefig(f"resultImages\\{name}_points.png")
        plt.show()
    
    ###Deuxième image, montrer la triangulation moyenne
    if (showFig):
        ###Première figure - montrer les points des deux images
        fig, ax = plt.subplots(1, 3, figsize=(12,4))

        # ---- First image ----
        ax[0].imshow(img1, extent=[0,720,0,720], origin='lower')
        ax[0].triplot(img1_pts[:,0], img1_pts[:,1], tri.simplices)
        ax[0].scatter(img1_pts[:,0], img1_pts[:,1],
              c=np.arange(len(img1_pts)),
              cmap='tab20',
              s=40)
        ax[0].set_xlim(0,720)
        ax[0].set_ylim(720,0)
        ax[0].set_title("Image 1")



        img1_5_pts = (img1_pts + img2_pts) / 2
        avg_img = (img1.astype(float) + img2.astype(float)) / (2*255)
        ax[1].imshow(avg_img, extent=[0,720,0,720], origin='lower')
        ax[1].triplot(img1_5_pts[:,0], img1_5_pts[:,1], tri.simplices)
        ax[1].scatter(img1_5_pts[:,0], img1_5_pts[:,1],
              c=np.arange(len(img1_5_pts)),
              cmap='tab20',
              s=40)
        ax[1].set_xlim(0,720)
        ax[1].set_ylim(720,0)
        ax[1].set_title("Image moyenne")

        # ---- Second image ----
        ax[2].imshow(img2, extent=[0,720,0,720], origin='lower')
        ax[2].triplot(img2_pts[:,0], img2_pts[:,1], tri.simplices)
        ax[2].scatter(img2_pts[:,0], img2_pts[:,1],
              c=np.arange(len(img2_pts)),
              cmap='tab20',
              s=40)
        ax[2].set_xlim(0,720)
        ax[2].set_ylim(720,0)
        ax[2].set_title("Image 2")

        plt.tight_layout()
        if (saveFig):
            plt.savefig(f"resultImages\\{name}_tri.png")
        plt.show()
    

    ###Get new points
    merge_pts = img1_pts + warp_frac* (img2_pts-img1_pts)

    ###Pour chaque triangle, trouver la transformation Affine et la stocker dans l'array
    Affine_transforms_simplices = []

    for p1, p2, p3 in tri.simplices[:2]:
        x1, y1 = img1_pts[p1]
        x2, y2 = img1_pts[p2]
        x3, y3 = img1_pts[p3]
        x1p, y1p = merge_pts[p1]
        x2p, y2p = merge_pts[p2]
        x3p, y3p = merge_pts[p3]

        ###  A Start [p1, p2, p3]  = End [p1p p2p p3p]
        ### A = E S^-1
        S = np.array([
            [x1, x2, x3],
            [y1, y2, y3],
            [1, 1, 1]
        ])

        E = np.array([
            [x1p, x2p, x3p],
            [y1p, y2p, y3p],
            [1, 1, 1]
        ])
        A = np.dot(E, np.linalg.inv(S))
        print(A)

        Affine_transforms_simplices.append(A)

    ### testing transform 

    ###Create a plot with a triangle from img1_pts , the same triangle form img2_pts, then compute with A the triangle strating from img1_pts
    ### homogeneous coordinates
    ### testing transform for this triangle
    p1, p2, p3 = tri.simplices[1]

    points_img1 = [img1_pts[p1], img1_pts[p2], img1_pts[p3]]
    points_img2 = [img2_pts[p1], img2_pts[p2], img2_pts[p3]]

    points_affine = []
    for point in points_img1:
        p = np.array([point[0], point[1], 1])  # homogeneous coord
        newPoint = np.dot(Affine_transforms_simplices[1], p)
        points_affine.append(newPoint[:2])

    fig, ax = plt.subplots()

    # img1 triangle
    for x, y in points_img1:
        ax.scatter(x, y, color='blue', label="img1")

    # img2 triangle
    for x, y in points_img2:
        ax.scatter(x, y, color='green', label='img2')

    # transformed points
    for x, y in points_affine:
        ax.scatter(x, y, color='red', label='imgMorph')

    ax.set_aspect('equal')
    ax.legend()
    ax.invert_yaxis()  # useful for image coordinates
    plt.show()



    return




tri = calculate_mean_tri(img1_pts, img2_pts)
morphed_img = morph(img1, img2, img1_pts, img2_pts, tri, showFig=False, saveFig=True)