import numpy as np
import pandas as pd
import scipy.io
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf
from pyriemann.utils.mean import mean_riemann
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann, distance_logeuclid, distance_euclid
from pyriemann.estimation import Covariances

# Load the concatenated dataset
file_name = 'S2_all_runs_concatenated.mat' 
data = scipy.io.loadmat(file_name)['run_data']
print("loaded data")

# Define parameters
window_size = 200  
num_channels = 256
trigger_row = 256  

# Get total number of timepoints
num_timepoints = data.shape[1]
num_epochs = num_timepoints // window_size

# Initialize storage for epochs and labels
epochs = np.zeros((num_epochs, num_channels, window_size))
epoch_labels = np.zeros(num_epochs, dtype=int)

# Extract epochs and assign labels
for i in range(num_epochs):
    start_idx = i * window_size
    end_idx = start_idx + window_size

    # Ensure data is sliced correctly and trigger data is cast to integers
    epochs[i] = data[:num_channels, start_idx:end_idx]
    trigger_data = data[trigger_row, start_idx:end_idx].astype(int)  # Cast to int
    if len(trigger_data) > 0:
        epoch_labels[i] = np.bincount(trigger_data).argmax()

# Group by class and compute average EEG activity for each class
unique_classes = np.unique(epoch_labels)
class_epochs = {}
average_class_epochs = {}

for class_id in unique_classes:
    mask = epoch_labels == class_id
    class_epochs[class_id] = epochs[mask]
    average_class_epochs[class_id] = np.mean(class_epochs[class_id], axis=0)

# Reorder dimensions for covariance calculation
epochs_permuted = np.transpose(epochs, (0, 2, 1))

# Calculate covariance matrices using Ledoit-Wolf estimator
cov_estimator = Covariances(estimator='oas')
cov_matrices = cov_estimator.fit_transform(epochs_permuted)

# Compute Riemannian means
mean_cov_matrices = {}
for class_id in unique_classes:
    if class_id not in [0, 7]:  # Exclude classes 0 and 7
        print(class_id)
        class_covs = cov_matrices[epoch_labels == class_id]
        mean_cov_matrices[class_id] = mean_riemann(class_covs)

print('Completed computing Riemannian mean covariance matrices for each class.')

distances = {}

# Compute distances for each class and store the results
for class_id in unique_classes:
    if class_id not in [0, 7]:  # Skip classes 0 and 7
        print(class_id)
        mean_cov_class = mean_cov_matrices[class_id]
        num_covs = cov_matrices.shape[0]
        distances_riemann = np.zeros(num_covs)
        distances_logeuclid = np.zeros(num_covs)
        distances_euclid = np.zeros(num_covs)

        for j in range(num_covs):
            cov_matrix = cov_matrices[j]
            print(j)
            # Calculate distances
            distances_riemann[j] = distance_riemann(cov_matrix, mean_cov_class)
            distances_logeuclid[j] = distance_logeuclid(cov_matrix, mean_cov_class)
            distances_euclid[j] = distance_euclid(cov_matrix, mean_cov_class)

        # Store distances 
        distances[class_id] = {
            'Riemann': distances_riemann,
            'LogEuclid': distances_logeuclid,
            'Euclidean': distances_euclid
        }

        print(f'Completed distance calculations for class {class_id}.')

# Storing Data
# Prepare a DataFrame to store all features and labels
all_features = []

for class_id in unique_classes:
    if class_id not in [0, 7]:  # Skip 0 and 7 
        print(class_id)
        num_total_epochs = len(distances[class_id]['Riemann'])
        class_label = np.full(num_total_epochs, class_id)

        # Access distance data for the current class
        Riemann_distances = distances[class_id]['Riemann']
        LogEuclid_distances = distances[class_id]['LogEuclid']
        Euclidean_distances = distances[class_id]['Euclidean']

        # Combine distances into a DataFrame
        class_features = pd.DataFrame({
            'Riemann': Riemann_distances,
            'LogEuclid': LogEuclid_distances,
            'Euclidean': Euclidean_distances,
            'Label': class_label
        })
        all_features.append(class_features)

# Concatenate all data into one DataFrame
features_table = pd.concat(all_features, ignore_index=True)

# Save to a CSV file 
features_table.to_csv('EEG_Features_Labels.csv', index=False)
print('Feature table processing complete for all classes.')