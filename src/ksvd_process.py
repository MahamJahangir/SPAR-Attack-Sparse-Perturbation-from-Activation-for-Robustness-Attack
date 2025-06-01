
import os
import numpy as np
from PIL import Image, ImageOps
from ksvd import ApproximateKSVD

os.makedirs("output/sparse_perturbations", exist_ok=True)

def apply_ksvd(perturbations):
    X = perturbations.reshape(len(perturbations), -1)
    aksvd = ApproximateKSVD(n_components=81)
    aksvd.fit(X)
    return aksvd.transform(X)

def save_sparse_images(gamma):
    for i, x in enumerate(gamma, 1):
        arr_9x9 = x.reshape(9, 9)
        norm_arr = 255 * (arr_9x9 - arr_9x9.min()) / (arr_9x9.max() - arr_9x9.min() + 1e-8)
        norm_arr = norm_arr.astype(np.uint8)
        pimg = Image.fromarray(norm_arr).convert("RGB")
        temp_path = f"output/sparse_perturbations/temp_{i}.png"
        pimg.save(temp_path)
        final_img = ImageOps.fit(Image.open(temp_path), (28, 28), Image.ANTIALIAS)
        final_img.save(f"output/sparse_perturbations/sparse_perturbation_{i}.png")
        os.remove(temp_path)
