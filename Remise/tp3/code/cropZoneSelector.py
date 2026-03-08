
""" MADE BY CHAT FOR CONVINIENCE """


import os
import glob
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# ============================================================
# CONFIG — make this match your pipeline
# ============================================================

# Use the SAME extensions list you use in your main code (same order!)
extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.JPG", "*.PNG"]

WINDOW_NAME = "Draw ROI | ENTER/SPACE=confirm | ESC=skip"
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 900


# ============================================================
# Helpers
# ============================================================

def norm_path(p: str) -> str:
    """Stable key for Windows paths."""
    return os.path.normcase(os.path.normpath(p))

def imread_unicode(path: str):
    """Reliable read for Windows paths with accents/OneDrive."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def resize_to_fit_screen(img):
    h, w = img.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)  # never upscale
    if scale == 1.0:
        return img, 1.0
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return disp, scale

def pick_input_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select images_a_accentuer_input_dir")
    root.destroy()
    return folder

def build_image_paths(images_a_accentuer_input_dir: str):
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(images_a_accentuer_input_dir, ext)))
    return image_paths


# ============================================================
# Main
# ============================================================

def main():
    images_a_accentuer_input_dir = pick_input_folder()
    if not images_a_accentuer_input_dir:
        print("No folder selected. Exiting.")
        return

    image_paths = build_image_paths(images_a_accentuer_input_dir)
    if not image_paths:
        print("No images found.")
        return

    # ROIs keyed by normalized full path
    crop_roi_by_path = {}

    print(f"\nFound {len(image_paths)} images.")
    print("For each image: draw ROI, ENTER/SPACE to confirm, ESC to skip.\n")

    for i, path in enumerate(image_paths, start=1):
        path = os.path.normpath(path)
        key = norm_path(path)
        filename = os.path.basename(path)

        img = imread_unicode(path)
        if img is None:
            print(f"[{i}/{len(image_paths)}] ❌ Could not read: {filename} -> skipped")
            crop_roi_by_path[key] = None
            continue

        disp, scale = resize_to_fit_screen(img)

        roi = cv2.selectROI(f"{WINDOW_NAME} ({i}/{len(image_paths)}) {filename}", disp,
                            showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()

        x, y, w, h = map(int, roi)
        if w == 0 or h == 0:
            print(f"[{i}/{len(image_paths)}] {filename}: skipped -> None")
            crop_roi_by_path[key] = None
            continue

        # Convert ROI back to original pixels
        x0 = int(x / scale)
        y0 = int(y / scale)
        w0 = int(w / scale)
        h0 = int(h / scale)

        crop_roi_by_path[key] = (x0, y0, w0, h0)
        print(f"[{i}/{len(image_paths)}] {filename}: ROI = ({x0}, {y0}, {w0}, {h0})")

    # Print copy/paste output
    print("\n================ COPY/PASTE THIS =================\n")
    print("import os\n")
    print("def _norm_path(p):")
    print("    return os.path.normcase(os.path.normpath(p))\n")

    print("crop_roi_by_path = {")
    for k, v in crop_roi_by_path.items():
        print(f"    {k!r}: {v},")
    print("}\n")
    print("# Use: roi = crop_roi_by_path.get(_norm_path(img_path))\n")
    print("==================================================\n")


if __name__ == "__main__":
    main()
