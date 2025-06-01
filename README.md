
# K-SVD MNIST Project

This repository contains a complete implementation of a pipeline using a Convolutional Neural Network (CNN) on the MNIST dataset and applying K-SVD for sparse representation of deconvolved perturbations.

## Structure

- `src/`: Source code organized into modular scripts.
- `output/`: Contains generated perturbations and sparse perturbation images.
- `models/`: Stores trained model weights.
- `main.py`: Main script to run the entire pipeline.

## Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib
- scikit-image
- Pillow
- ksvd (custom or installable package)

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Ensure the `ksvd` package is installed or available in your path.
