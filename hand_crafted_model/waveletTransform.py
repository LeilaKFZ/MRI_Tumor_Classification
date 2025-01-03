import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
import pywt


# ------------------------------
# PREPROCESSING FUNCTIONS
# ------------------------------

def preprocess_image(image_path):
    """Preprocess img to detect anomalies"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"L'image {image_path} n'a pas pu être chargée.")

    # noise reduction
    filtered_image = cv2.GaussianBlur(image, (7, 7), 2)

    # 2D Wavelet Transform
    coeffs2 = pywt.dwt2(filtered_image, 'bior1.3')
    _, (LH, HL, HH) = coeffs2

    # Combine all the coeff to detect edges
    contours = np.abs(LH) + np.abs(HL) + np.abs(HH)

    threshold = 10
    suspect_image = (contours > threshold).astype(np.uint8)

    circles = cv2.HoughCircles(filtered_image, cv2.HOUGH_GRADIENT, dp=1.5, minDist=150,
                               param1=100, param2=40, minRadius=60, maxRadius=130)

    # Si des cercles sont détectés, les dessiner pour les exclure
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, radius = circle
            # Dessiner le cercle uniquement en bordure (contour du crâne)
            cv2.circle(suspect_image, (x, y), radius + 10, 0, 2)

    return suspect_image


def classify_image(suspect_image, threshold_decision=0.2):
    """Classifie une image comme ayant une tumeur ou non en fonction du ratio de pixels suspects."""
    tumor_pixels = np.sum(suspect_image > 0)
    total_pixels = suspect_image.size
    tumor_ratio = tumor_pixels / total_pixels
    return tumor_ratio > threshold_decision  # True pour tumeur, False sinon


# ------------------------------
# EVALUATION Functions
# ------------------------------

def calculate_metrics(true_labels, predicted_labels):
    """Calcule les métriques de classification."""
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label=1)
    recall = recall_score(true_labels, predicted_labels, pos_label=1)
    f1 = f1_score(true_labels, predicted_labels, pos_label=1)
    return accuracy, precision, recall, f1


def evaluate_and_display_metrics(labels, predictions, dataset_name="Test"):
    """Calcule et affiche les métriques ainsi que la matrice de confusion."""
    accuracy, precision, recall, f1 = calculate_metrics(labels, predictions)

    print(f"\n=== Résultats {dataset_name} ===")
    print(f"Accuracy : {accuracy:.2%}")
    print(f"Precision : {precision:.2%}")
    print(f"Recall : {recall:.2%}")
    print(f"F1-score : {f1:.2%}")

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Tumeur", "Tumeur"])
    disp.plot(cmap="Blues")
    plt.show()


# ------------------------------
# DATA MANAGEMENT
# ------------------------------

def prepare_data(tumor_dir, no_tumor_dir, train_dir, test_dir, test_size=0.3):
    """Prépare les ensembles d'entraînement et de test."""
    tumor_files = [(os.path.join(tumor_dir, f), 1) for f in os.listdir(tumor_dir)]
    no_tumor_files = [(os.path.join(no_tumor_dir, f), 0) for f in os.listdir(no_tumor_dir)]

    # Fusionner et mélanger
    all_data = tumor_files + no_tumor_files
    random.shuffle(all_data)

    # Division des données
    train_data, test_data = train_test_split(all_data, test_size=test_size, random_state=42)

    # Organisation des fichiers
    for path, label in train_data:
        label_dir = os.path.join(train_dir, "tumor" if label == 1 else "no_tumor")
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(path, label_dir)

    # Fusionner les données de test
    os.makedirs(test_dir, exist_ok=True)
    for i, (path, label) in enumerate(test_data):
        new_filename = f"test_{i}_{'tumor' if label == 1 else 'no_tumor'}.png"
        shutil.copy(path, os.path.join(test_dir, new_filename))

    return train_data, test_data


# ------------------------------
# PRINCIPAL PIPELINE
# ------------------------------

def main():
    # Répertoires
    no_tumor_dir = "../Datas/imgResized/no_tumor"
    tumor_dir = "../Datas/imgResized/tumor"
    train_dir = "../Datas/train_dataset"
    test_dir = "../Datas/test_combined_dataset"

    # Préparation des données
    train_data, test_data = prepare_data(tumor_dir, no_tumor_dir, train_dir, test_dir)

    # Classification des données d'entraînement
    train_labels, train_predictions = [], []
    for path, label in train_data:
        suspect_image = preprocess_image(path)
        predicted_label = classify_image(suspect_image)
        train_labels.append(label)
        train_predictions.append(1 if predicted_label else 0)

    # Évaluation des données d'entraînement
    evaluate_and_display_metrics(train_labels, train_predictions, dataset_name="Entraînement")

    # Classification des données de test
    test_labels, test_predictions = [], []
    for path, label in test_data:
        suspect_image = preprocess_image(path)
        predicted_label = classify_image(suspect_image)
        test_labels.append(label)
        test_predictions.append(1 if predicted_label else 0)

    # Évaluation des données de test
    evaluate_and_display_metrics(test_labels, test_predictions, dataset_name="Test")


if __name__ == "__main__":
    main()
