import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

plt.ion()  # Turns on interactive mode for matplotlib 

# Load from CSV file
features_table = pd.read_csv('EEG_Features_Labels.csv')
print('loaded data')

print("Feature Summary Statistics by Class:")
print(features_table.groupby("Label").describe())

# Access features and labels
features = features_table[['Riemann', 'LogEuclid', 'Euclidean']]
labels = features_table['Label']
print('accessed features')

# Standardizing the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=16, learning_rate=200, n_iter=1000, method='barnes_hut')
tsne_features = tsne.fit_transform(features)

# Plot t-SNE results
plt.figure()
plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='viridis')
plt.title('Adjusted t-SNE Visualization of EEG Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar()
plt.savefig('tSNE_Visualization.png', format='png', dpi=300)
plt.show(block=False)

print("performing LDA")
lda = LDA(n_components=2)
lda_features = lda.fit_transform(features, labels)

plt.scatter(lda_features[:, 0], lda_features[:, 1], c=labels, cmap='viridis')
plt.title('LDA Visualization of EEG Features')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.colorbar()
plt.savefig('LDA_Feature_Visualization.png', format='png', dpi=300)
plt.show(block=False)

# Cross-validate the LDA model
print("Starting cross-validation...")
cv_scores = cross_val_score(lda, features, labels, cv=5)
validation_accuracy = np.mean(cv_scores)
print(f'Validation accuracy: {validation_accuracy * 100:.2f}%')

# Use LDA for dimension reduction
print("Fitting LDA model for dimension reduction...")
lda.fit(features, labels)
lda_features = lda.transform(features)

# Visualize the first two LDA dimensions
print("Visualizing LDA reduced dimensions...")
plt.figure()
plt.scatter(lda_features[:, 0], lda_features[:, 1], c=labels, cmap='viridis')
plt.title('LDA Reduced Dimension Visualization of EEG Features')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.colorbar()
plt.savefig('LDA_Visualization.png', format='png', dpi=300)
plt.show(block=False)

# Euclidean vs Riemann
print("Applying LDA to Euclidean features...")
euclidean_features = features['Euclidean'].values.reshape(-1, 1)
lda_euclidean = LDA().fit(euclidean_features, labels)
lda_euclidean_features = lda_euclidean.transform(euclidean_features)

print("Applying LDA to Riemann features...")
riemann_features = features['Riemann'].values.reshape(-1, 1)
lda_riemann = LDA().fit(riemann_features, labels)
lda_riemann_features = lda_riemann.transform(riemann_features)

# Plotting
print("Plotting the results for both feature sets...")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(lda_euclidean_features, np.zeros_like(lda_euclidean_features), c=labels, cmap='viridis')
plt.title('LDA of Euclidean Features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.subplot(1, 2, 2)
plt.scatter(lda_riemann_features, np.zeros_like(lda_riemann_features), c=labels, cmap='viridis')
plt.title('LDA of Riemann Features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.tight_layout()
plt.show(block=False)