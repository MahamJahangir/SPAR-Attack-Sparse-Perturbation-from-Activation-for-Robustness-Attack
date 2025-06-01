
from src.data_loader import load_data
from src.model import build_and_train_model, load_model_weights
from src.perturbation import generate_perturbations
from src.ksvd_process import apply_ksvd, save_sparse_images
from src.eval import evaluate_adversarial

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Build and train model
    model = build_and_train_model(X_train, y_train, X_test, y_test)

    # Load model weights
    model = load_model_weights(model)

    # Generate perturbations
    perturbations = generate_perturbations(model, X_test)

    # Apply K-SVD and save sparse perturbations
    gamma = apply_ksvd(perturbations)
    save_sparse_images(gamma)

    # Evaluate the effect of perturbations
    evaluate_adversarial(model, X_test, y_test)

if __name__ == "__main__":
    main()
