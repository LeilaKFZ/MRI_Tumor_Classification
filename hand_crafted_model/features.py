import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


class ImageSegment:
    """
    Represents a segment of an img, includinf its metadata and content
    """
    def __init__(self, segment, position, image_path):
        self.segment = segment
        self.position = position
        self.image_path = image_path
        self.size_bytes = sys.getsizeof(segment)
        self.histogram = None  # Histogramme initialisé à None


class ImageProcessor:
    """
    Segmentation and Histogram calculation
    """
    def __init__(self, image, image_path):
        if not isinstance(image, np.ndarray):
            raise ValueError("L'image doit être un numpy.ndarray.")
        self.image = image
        self.image_path = image_path
        self.segments = []

    def split(self, m, n, plot=False):
        """
        Divise l'image en segments et les stocke.

        :param m: nbr of segment on height
        :param n: nbr of segment on weight
        :param plot: display each segment if true
        """
        if not isinstance(m, int) or m <= 0:
            raise ValueError("Le paramètre 'm' doit être un entier positif.")
        if not isinstance(n, int) or n <= 0:
            raise ValueError("Le paramètre 'n' doit être un entier positif.")

        h, w = self.image.shape
        segment_h, segment_w = h // m, w // n

        for i in range(m):
            for j in range(n):
                x_start, x_end = i * segment_h, (i + 1) * segment_h
                y_start, y_end = j * segment_w, (j + 1) * segment_w
                segment = self.image[x_start:x_end, y_start:y_end]
                self.segments.append(ImageSegment(segment, (i, j), self.image_path))

                if plot:
                    plt.imshow(segment, cmap='gray')
                    plt.title(f"Région ({i}, {j})\n{self.image_path}")
                    plt.axis('off')
                    plt.show()

    def compute_histograms(self, i_range):
        """
        Calculates histograms for each segment
        :param i_range: Range interval for grayscale levels
        """
        if not isinstance(i_range, int) or i_range <= 0:
            raise ValueError("Le paramètre 'i_range' doit être un entier positif.")

        num_intervals = 256 // i_range  # Nombre de plages

        for segment_data in self.segments:
            histogram = [0] * num_intervals  # Initialise un histogramme avec toutes les fréquences à 0
            for row in segment_data.segment:
                for pixel in row:
                    bin_index = pixel // i_range  # Trouver la plage correspondant au pixel
                    histogram[bin_index] += 1  # Incrémenter la fréquence de la plage
            segment_data.histogram = histogram  # Stocker l'histogramme dans le segment

    def plot_histograms(self, i_range):
        """
        Plots histograms of the segments with manually calculated ranges
        :param i_range: Interval used to define grayscale level ranges
        """
        if not self.segments:
            raise ValueError("Aucun segment disponible. Veuillez d'abord segmenter l'image.")

        if any(seg.histogram is None for seg in self.segments):
            raise ValueError("Les histogrammes ne sont pas calculés. Appelez 'compute_histograms()' d'abord.")

        num_intervals = 256 // i_range
        bin_edges = [i * i_range for i in range(num_intervals + 1)]  # Bords des intervalles
        x_labels = [f"{bin_edges[i]}-{bin_edges[i + 1] - 1}" for i in range(num_intervals)]

        # Utilisation de la palette tab20 pour des couleurs lisibles
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.segments)))

        plt.figure(figsize=(14, 8))

        for idx, segment in enumerate(self.segments):
            x = np.arange(num_intervals)  # Indices des plages
            plt.bar(
                x, segment.histogram,
                width=0.8, color=colors[idx % len(colors)], alpha=0.8,
                label=f"Segment {segment.position}"
            )

        # Add labels
        plt.xticks(np.arange(num_intervals), x_labels, rotation=45, ha='right')
        plt.xlabel(f'Plages de niveaux de gris (i_range={i_range})')
        plt.ylabel('Fréquence')
        plt.title(f"Histogrammes des segments (calcul manuel)\nChemin de l'image : {self.image_path}")
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.show()


def process_brain_images(folder_path, m, n, i_range, output_file):
    """
        Complete process to segment images, calculate histograms, and save them.

        :param folder_path: Path of the folder containing the images.
        :param m: Number of segments in height.
        :param n: Number of segments in width.
        :param i_range: Interval for grayscale level ranges.
        :param output_file: Output file to save the results.
    """

    if not os.path.exists(folder_path):
        raise ValueError(f"Le dossier {folder_path} n'existe pas.")

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not image_files:
        raise ValueError("Aucune image trouvée dans le dossier spécifié.")

    results = {}

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Erreur lors du chargement de l'image : {image_path}")
            continue

        processor = ImageProcessor(image, image_path)

        # Étape 1 : Segmentation
        processor.split(m, n, plot=False)

        # Étape 2 : Calcul des histogrammes
        processor.compute_histograms(i_range)

        # Sauvegarder les résultats
        results[image_path] = [
            {
                'position': seg.position,
                'histogram': seg.histogram,
                'size_bytes': seg.size_bytes,
            }
            for seg in processor.segments
        ]

    # Sauvegarder toutes les données dans un fichier unique
    np.save(output_file, results)
    print(f"Les données cumulées ont été sauvegardées dans {output_file}.")


if __name__ == "__main__":
    folder_path = "../Datas/tum_observation"
    m, n = 3, 4  # lignes, colonnes
    i_range = 16

    # Parcourir les images du dossier et afficher les histogrammes
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            processor = ImageProcessor(image, image_path)

            # Étape 1 : Segmentation
            processor.split(m, n, plot=False)

            # Étape 2 : Calcul des histogrammes
            processor.compute_histograms(i_range)

            # Étape 3 : Affichage des histogrammes
            processor.plot_histograms(i_range)
        else:
            print(f"Impossible de charger l'image : {image_path}")
