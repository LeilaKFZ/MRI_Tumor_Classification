import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt


# Détection de l'appareil (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classe personnalisée pour charger les images et leurs labels
class BrainTumorDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Répertoires des classes
no_tumor_dir = "../Datas/imgResized/no_tumor"
tumor_dir = "../Datas/imgResized/tumor"

# Créer les listes de fichiers et labels
no_tumor_files = [os.path.join(no_tumor_dir, f) for f in os.listdir(no_tumor_dir) if f.endswith(('.jpg', '.png'))]
tumor_files = [os.path.join(tumor_dir, f) for f in os.listdir(tumor_dir) if f.endswith(('.jpg', '.png'))]

# Associer les labels : 0 pour no_tumor, 1 pour tumor
file_paths = no_tumor_files + tumor_files
labels = [0] * len(no_tumor_files) + [1] * len(tumor_files)

# Diviser en 80% train et 20% temporaire (validation + test)
train_files, temp_files, train_labels, temp_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Diviser temporaire en 50% validation et 50% test
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Nombre d'images d'entraînement : {len(train_files)}")
print(f"Nombre d'images de validation : {len(val_files)}")
print(f"Nombre d'images de test : {len(test_files)}")

# Définir les transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalisation RGB
])

# Créer les datasets et dataloaders
train_dataset = BrainTumorDataset(train_files, train_labels, transform=data_transforms)
val_dataset = BrainTumorDataset(val_files, val_labels, transform=data_transforms)
test_dataset = BrainTumorDataset(test_files, test_labels, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# Chargement du modèle ResNet
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Fonction pour entraîner le modèle avec suivi des courbes et métriques d'apprentissage 

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}  # Historique des pertes et exactitudes
    metrics_history = []  # Historique des métriques (précision, rappel, F1, matrice de confusion)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Boucle à travers les données
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Collecte des prédictions et labels uniquement pour la validation
                if phase == 'val':
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Sauvegarder les valeurs dans l'historique
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                # Calculer les métriques et la matrice de confusion pour la validation
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                precision = precision_score(all_labels, all_preds, average='binary')
                recall = recall_score(all_labels, all_preds, average='binary')
                f1 = f1_score(all_labels, all_preds, average='binary')
                conf_matrix = confusion_matrix(all_labels, all_preds)

                # Sauvegarder dans l'historique
                metrics_history.append({
                    'epoch': epoch + 1,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'conf_matrix': conf_matrix
                })

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        print()

    # Tracer les courbes d'apprentissage 
    plot_learning_curves(history)

    # Afficher les métriques et matrices de confusion 
    display_metrics(metrics_history)

    return model

#  Fonction pour tracer les courbes d'apprentissage
def plot_learning_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)

    # Tracer les pertes
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Courbe de la perte')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.legend()

    # Tracer les précisions
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Courbe de l\'exactitude')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Fonction pour afficher les métriques et matrices de confusion
def display_metrics(metrics_history):
    print("\nMétriques et matrices de confusion pour chaque époque :\n")
    for entry in metrics_history:
        print(f"Époque {entry['epoch']}:")
        print(f"  Précision : {entry['precision']:.4f}")
        print(f"  Rappel : {entry['recall']:.4f}")
        print(f"  F1-score : {entry['f1']:.4f}")
        print(f"  Matrice de confusion :\n{entry['conf_matrix']}\n")

############## Entraîner le modèle ##############################################################################
model = train_model(model, criterion, optimizer, scheduler, num_epochs=10)
