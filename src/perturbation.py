
import os
import numpy as np
from skimage import restoration
from skimage.transform import resize
from tensorflow.keras.models import Model
from matplotlib.image import imsave

os.makedirs("output/perturbations", exist_ok=True)

def generate_perturbations(model, X_test):
    sweights = model.layers[1].get_weights()[0]
    wts = sweights[:, :, 5, 1]  # example filter index

    activation_model = Model(inputs=model.input, outputs=[model.layers[1].output])
    perturbations = []

    for i in range(len(X_test)):
        x = X_test[i:i+1]
        activations = activation_model.predict(x, verbose=0)
        activation_map = activations[0, :, :, 1]

        psf = wts.reshape(3, 3)
        deconvolved, _ = restoration.unsupervised_wiener(activation_map, psf)
        deconvolved_resized = resize(deconvolved, (28, 28), preserve_range=True)
        deconvolved_resized = deconvolved_resized.reshape(1, 28, 28, 1)

        perturbation = deconvolved_resized - x
        perturbations.append(perturbation)

        pert_img = perturbation.squeeze()
        pert_img_normalized = (pert_img - pert_img.min()) / (pert_img.max() - pert_img.min() + 1e-8)
        imsave(f"output/perturbations/perturbation_{i}.png", pert_img_normalized, cmap='gray')
        np.save(f"output/perturbations/perturbation_{i}.npy", perturbation)

    return np.array(perturbations).reshape(-1, 28, 28, 1)
