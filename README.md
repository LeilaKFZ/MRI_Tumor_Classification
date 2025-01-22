# Binary Classification of Brain MRI Images
## Project Overview
This project focuses on the binary classification of brain MRI images to distinguish between images with tumors and those without. The classification is performed using three approaches: Hand-crafted features, Machine Learning (ML), and Deep Learning (DL). The dataset was preprocessed to remove low-quality images and those containing irrelevant regions (e.g., eyes or full skull), resulting in 4,782 images without tumors and 3,846 images with tumors. The dataset was compiled from multiple sources, which introduced some redundancy due to repeated images.
## Dataset
•	Initial Dataset: Compiled from multiple sources, leading to some redundancy (repeated images with different labels).
•	Preprocessing:
o	Removed low-quality images and those containing irrelevant regions (e.g., eyes or full skull).
o	Images were converted to grayscale and resized (dimensions varied by method).
•	Final Dataset:
o	No Tumor: 4,782 images.
o	Tumor: 3,846 images.
# Methods
## 1. Hand-Crafted Models (Experimental)
The hand-crafted methods were implemented as an experimental exercise to explore traditional statistical and image processing techniques. These methods were used to gain a deeper understanding of the dataset and to experiment with direct metric-based approaches before moving on to more advanced ML and DL techniques.
### Method 1: Histogram-based Analysis
•	Metrics: Standard deviation and correlation to identify tumor regions.
•	Results: Achieved 71.44% accuracy with a 2x2 segmentation.
### Method 2: Wavelet Transform
•	Approach: Used 2D wavelet transform and HOG to detect edges and exclude skull structures.
•	Results: Achieved 50.29% accuracy.
Here is a diagram that summerizes all the method's steps:
![Image](https://github.com/user-attachments/assets/8610b669-c940-4c1d-b236-9a58b4d88233)

## 2. Machine Learning (ML)
### Method: PCA + SVM
•	Preprocessing: Images resized to 128x128, normalized, and reduced using PCA (50 components).
•	Model: SVM with C=1.0 and gamma='scale'.
•	Results: Achieved 98% accuracy on the test set.
## 3. Deep Learning (DL)
### Method: Pre-trained ResNet-18
•	Preprocessing: Images resized to 64x64 and converted to tensors.
•	Model: ResNet-18 pre-trained on ImageNet, modified for binary classification.
•	Hyperparameters: Adam optimizer, learning rate=0.001, 25 epochs, batch size=16.
•	Results: Achieved 100% accuracy on the test set with early stopping at epoch 12.
Conclusion
•	Hand-Crafted Features: These methods were implemented as an experimental exercise to explore traditional techniques. They achieved moderate accuracy (71.44% for histogram-based analysis and 50.29% for wavelet transform).
•	Machine Learning: High accuracy (98%) with PCA + SVM.
•	Deep Learning: Best performance (100%) with ResNet-18.

