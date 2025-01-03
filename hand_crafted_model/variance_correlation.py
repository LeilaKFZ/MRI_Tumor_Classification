import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

# Load (NxM).npy files
dataNoTum = np.load('../Datas/hist_2x2_noTumor.npy', allow_pickle=True).item()
dataTum = np.load('../Datas/hist_2x2_tumor.npy', allow_pickle=True).item()


# Function to filter out black regions
def filter_black_regions(histograms, threshold=6000):
    """
    Filters out histograms where the sum of values is below a given threshold.
    """
    return [hist for hist in histograms if np.sum(hist) > threshold]


# Function to classify images
def classify_image(std_devs, correlations, std_threshold= 5000, corr_threshold=0.7):
    # Step 1: Variability (standard deviation)
    high_variability = any(std > std_threshold for std in std_devs)
    if not high_variability:
        return "NONtumor"  # No significant variability

    # Step 2: Validation by correlation
    low_correlation = any(corr[2] < corr_threshold for corr in correlations)
    if low_correlation:
        return "tumor"  # Low correlation confirmed

    return "NONtumor"  # Default to NONtumor if variability but strong correlations



# Analyze histograms and classify
def analyze_data(data, true_label, std_devs_global):
    """
    Analyzes the data, classifies images, and collects standard deviations globally.
    """
    classifications = []
    true_labels = []
    for image_path, segments in data.items():
        print(f"\nProcessing image: {image_path}")

        # Extract and filter histograms
        histograms = [segment['histogram'] for segment in segments]
        print(f"  Number of segments before filtering: {len(histograms)}")

        filtered_histograms = filter_black_regions(histograms, threshold=16000)
        print(f"  Number of segments after filtering: {len(filtered_histograms)}")

        if filtered_histograms:
            # Compute standard deviation for each histogram
            std_devs = [np.std(hist) for hist in filtered_histograms]
            print(f"  Standard deviations of filtered histograms: {std_devs}")

            # Add to global list for threshold calculation
            std_devs_global.extend(std_devs)

            # Compute correlations between histograms
            correlations = []
            num_histograms = len(filtered_histograms)
            for i in range(num_histograms):
                for j in range(i + 1, num_histograms):
                    try:
                        correlation = np.corrcoef(filtered_histograms[i], filtered_histograms[j])[0, 1]
                        correlations.append((i, j, correlation))
                    except Exception as e:
                        print(f"  Error calculating correlation: {e}")

            if correlations:
                print(f"  Correlations calculated: {[corr[2] for corr in correlations]}")
            else:
                print(f"  No correlations calculated.")

            # Classification
            classification = classify_image(std_devs, correlations)
            print(f"  Classification result: {classification}")

            classifications.append(classification)
            true_labels.append(true_label)

    return classifications, true_labels


# Global list for standard deviations
std_devs_global = []

# Analyze tumor and non-tumor data
print("\nAnalyzing NONtumor data...")
classifications_noTum, true_labels_noTum = analyze_data(dataNoTum, "NONtumor", std_devs_global)

print("\nAnalyzing tumor data...")
classifications_tum, true_labels_tum = analyze_data(dataTum, "tumor", std_devs_global)

# Combine results
all_classifications = classifications_noTum + classifications_tum
all_true_labels = true_labels_noTum + true_labels_tum

# Calculate and display global standard deviation statistics
mean_std_dev = np.mean(std_devs_global)
std_std_dev = np.std(std_devs_global)
print("\nGlobal Standard Deviation Analysis:")
print(f"  Mean of Standard Deviations: {mean_std_dev:.2f}")
print(f"  Standard Deviation of Standard Deviations: {std_std_dev:.2f}")

# Calculate accuracy
accuracy = accuracy_score(all_true_labels, all_classifications)
print("\nOverall Performance:")
print(f"  Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_classifications, labels=["NONtumor", "tumor"])
print("\nConfusion Matrix:")
print(conf_matrix)


'''
# Print results for each image
print("\nDetailed Results:")
for idx, (true_label, predicted) in enumerate(zip(all_true_labels, all_classifications)):
    print(f"Image {idx + 1}: True class = {true_label}, Predicted = {predicted}")
'''