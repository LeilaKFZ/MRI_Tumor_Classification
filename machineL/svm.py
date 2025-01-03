import numpy as np
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_files
from skimage.io import imread_collection
from skimage.color import rgb2gray
from skimage.transform import resize

brain_tumor_path = "../Datas/imgResized/tumor"
brain_noTumor_path = "../Datas/imgResized/no_tumor"

tumor_images_paths = glob.glob(brain_tumor_path)
no_tumor_images_paths = glob.glob(brain_noTumor_path)

# Charger les images et les étiquettes
tumor_images = imread_collection(brain_tumor_path)
no_tumor_images = imread_collection(brain_noTumor_path)

# Préparer les étiquettes : 1 pour "tumeur", 0 pour "pas de tumeur"
tumor_labels = [1] * len(tumor_images)
no_tumor_labels = [0] * len(no_tumor_images)

# Convertir les images en niveaux de gris et les redimensionner
def preprocess_images(image_collection, target_size=(64, 64)):
    processed_images = [
        resize(rgb2gray(image), target_size, anti_aliasing=True).flatten()
        for image in image_collection
    ]
    return np.array(processed_images)

X_tumor = preprocess_images(tumor_images)
X_no_tumor = preprocess_images(no_tumor_images)

# Fusionner les données et étiquettes
X = np.vstack((X_tumor, X_no_tumor))
y = np.hstack((tumor_labels, no_tumor_labels))

# Diviser les données en ensembles d'entraînement, validation et test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Définir la grille d'hyperparamètres
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Configurer et exécuter GridSearchCV
svm = SVC()
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres et le meilleur score
print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur score sur validation croisée :", grid_search.best_score_)

# Utiliser le modèle optimisé pour prédire sur l'ensemble de test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Évaluer les performances
print("\nAccuracy sur l'ensemble de test :", accuracy_score(y_test, y_pred))
print("\nRapport de classification:\n", classification_report(y_test, y_pred))
