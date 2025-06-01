
import numpy as np
import statistics
from PIL import Image
from tensorflow.keras.models import load_model

def evaluate_adversarial(model, X_test, y_test):
    print("Applying sparse perturbation and evaluating...")

    epsilon = 0.00004
    pert_image = Image.open("output/sparse_perturbations/sparse_perturbation_118.png").convert("L")
    p = np.array(pert_image).reshape(1, 28, 28, 1)

    x_noisy_image = []
    for x in X_test:
        x = x.reshape(1, 28, 28, 1)
        noisy_image = x + epsilon * p
        x_noisy_image.append(noisy_image)

    x_noisy_image = np.array(x_noisy_image).reshape(X_test.shape)

    norms = [np.linalg.norm(X_test[i] - x_noisy_image[i]) for i in range(len(X_test))]
    print("Mean L2 norm:", statistics.mean(norms))
    print("Median L2 norm:", statistics.median(norms))

    clean_acc = model.evaluate(X_test, y_test, verbose=0)
    adversarial_acc = model.evaluate(x_noisy_image, y_test, verbose=0)
    print("Clean Accuracy:", clean_acc)
    print("Adversarial Accuracy:", adversarial_acc)
