import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import scipy
import umap

from sklearn.metrics import accuracy_score
from autoencoder import get_data


base_directory = 'FaceRecognition/'
train_directory = base_directory + 'train'
test_directory = base_directory + 'test'


def load_and_vectorize(directory):
	image_vectors = []
	normalized_vectors = []
	labels = []

	for filename in os.listdir(directory):
		if filename.endswith(".png"):
			# Read in grayscale image
			image_path = os.path.join(directory, filename)
			image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

			# Vectorizing image and collecting label
			image_vector = image.flatten()
			image_vectors.append(image_vector)

			labels.append(int(filename.split("_")[0]))

	# Normalization and centering to be zero-mean
	image_vectors = np.array(image_vectors)
	mean = np.mean(image_vectors, axis=0)
	centered_data = image_vectors - mean

	normalized_data = centered_data / np.linalg.norm(centered_data, axis=1, keepdims=True)

	return normalized_data.T, labels

def pca(data, p):
	# Compute submatrix, computational trick
	submatrix = data.T @ data

	# Eignedecomposition of the submatrix
	sub_eigenvalues, sub_eigenvectors = np.linalg.eigh(submatrix)

	# Sort in descending order, choose eigenvectors that correspond to the p-largest eigenvalues
	descending_sorted_indices = np.argsort(sub_eigenvalues)[::-1]
	sub_eigenvectors = sub_eigenvectors[:, descending_sorted_indices][:, :p]

	# Mapping back to be true eigenvectors of the covariance matrix + nnormalization
	eigenvectors = data @ sub_eigenvectors
	eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

	return eigenvectors

def lda(data, labels, p):
	# Perform PCA for dimensionality reduction
	pca_eigenvectors = pca(data, p)
	reduced_data = pca_eigenvectors.T @ data

	# Compute within-class and between-class scatter matrices in PCA space

	# Calculating within-class and between-class scatter matrices
	S_W_reduced = np.zeros((reduced_data.shape[0], reduced_data.shape[0]))
	S_B_reduced = np.zeros((reduced_data.shape[0], reduced_data.shape[0]))


	unique_classes = np.unique(labels)
	overall_mean = np.mean(reduced_data, axis=1)

	# Calculation of within-class and between-class scatter
	for class_ in unique_classes:
		class_data = reduced_data[:, np.array(labels) == class_]
		class_mean = np.mean(class_data, axis=1)
		class_scatter = (class_data - class_mean[:, np.newaxis]) @ (class_data - class_mean[:, np.newaxis]).T
		S_W_reduced += class_scatter
		
		mean_difference = (class_mean - overall_mean).reshape(-1, 1)
		S_B_reduced += class_data.shape[1] * (mean_difference @ mean_difference.T)


	# Eigendecomposition and keeping the eigenvectors corresponding to the p-largest eigenvalues
	lda_eigenvalues, lda_eigenvectors = scipy.linalg.eigh(S_B_reduced, S_W_reduced + np.eye(S_W_reduced.shape[0]) * 1e-6)
	descending_sorted_indices = np.argsort(lda_eigenvalues)[::-1]
	lda_eigenvectors = lda_eigenvectors[:, descending_sorted_indices][:, :p]
	
	# Step 4: Map LDA eigenvectors back to original space + normalization
	lda_eigenvectors = pca_eigenvectors @ lda_eigenvectors
	lda_eigenvectors = lda_eigenvectors / np.linalg.norm(lda_eigenvectors, axis=0)
	

	return lda_eigenvectors

def project_to_subspace(data, pca_feature_set):
	return pca_feature_set.T @ data

def nearest_neighbor(train_projected, train_labels, test_projected):
	predictions = []

	for test_sample in test_projected.T:
		# Calculate L2 norm to all training samples
		distances = np.linalg.norm(train_projected.T - test_sample, axis=1)

		# Find the index of the closest training sample
		nearest_index = np.argmin(distances)

		# Assign label
		predictions.append(train_labels[nearest_index])

	return predictions


def plot_umap(train_data, train_labels, test_data, test_labels, predicted_labels, method, p):
	# Apply UMAP to reduce the data to 2D
	reducer = umap.UMAP(n_components=2, random_state=42)
	
	# Apply UMAP on training data
	train_umap = reducer.fit_transform(train_data.T)
	# Apply UMAP on test data
	test_umap = reducer.transform(test_data.T)
	
	plt.figure(figsize=(10, 5))
	
	# Plot training data
	plt.subplot(1, 2, 1)
	plt.scatter(train_umap[:, 0], train_umap[:, 1], c=train_labels, cmap='Spectral', edgecolors='k', s=40)
	plt.title(f"Training Data UMAP ({method})")
	plt.colorbar(label="Class")
	
	# Plot test data
	plt.subplot(1, 2, 2)
	plt.scatter(test_umap[:, 0], test_umap[:, 1], c=predicted_labels, cmap='Spectral', edgecolors='k', s=40)
	plt.title(f"Test Data UMAP ({method})")
	plt.colorbar(label="Predicted Class")
	
	plt.suptitle(f"UMAP Plot for {method} with Retaining {p} Eigenvectors", fontsize=16)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	train_data, train_labels = load_and_vectorize(train_directory)
	test_data, test_labels = load_and_vectorize(test_directory)

	print("train data shape: ", train_data.shape)

	time.sleep(100)

	p_set = range(1, 17)
	ae_p_set = [3, 8, 16]
	pca_accuracies = []
	lda_accuracies = []
	ae_accuracies = []

	plot_umaps = True

	for p in p_set:

		# Perform PCA

		# Collect feature set
		pca_feature_set = pca(train_data, p)
		# Project to PCA subspace
		pca_train_feature_vector = project_to_subspace(train_data, pca_feature_set)
		pca_test_feature_vector = project_to_subspace(test_data, pca_feature_set)
		# print("lda_train_feature_vector shape: ", pca_train_feature_vector.shape)
		# Predict labels with nearest neighbor algorithm
		pca_predicted_labels = nearest_neighbor(pca_train_feature_vector, train_labels, pca_test_feature_vector)
		# Calculate accuracy through #correct_predictions / #total_images
		pca_accuracy = accuracy_score(test_labels, pca_predicted_labels)


		# Perform LDA
		# Collect feature set
		lda_feature_set = lda(train_data, train_labels, p)
		# Project to LDA subspace
		lda_train_feature_vector = project_to_subspace(train_data, lda_feature_set)
		lda_test_feature_vector = project_to_subspace(test_data, lda_feature_set)
		# Predict labels with nearest neighbor algorithm
		lda_predicted_labels = nearest_neighbor(lda_train_feature_vector, train_labels, lda_test_feature_vector)
		# Calculate accuracy through #correct_predictions / #total_images
		lda_accuracy = accuracy_score(test_labels, lda_predicted_labels)

		pca_accuracies.append(pca_accuracy)
		lda_accuracies.append(lda_accuracy)

		print(f"p = {p}: PCA Accuracy = {pca_accuracy:.2%}, LDA Accuracy = {lda_accuracy:.2%}")

		if (p == 3 or p == 8 or p == 16):
			# Load autoencoder vectors and labels
			ae_train_feature_vector, ae_train_labels, ae_test_feature_vector, ae_test_labels = get_data(training=False, p=p)
			# Converting to be of shape (p, C) rather than (C, p)
			ae_train_feature_vector = ae_train_feature_vector.T
			ae_test_feature_vector = ae_test_feature_vector.T

			# Use nearest neighbors to predict test labels with autoencoder embeddings
			ae_predicted_labels = nearest_neighbor(ae_train_feature_vector, ae_train_labels, ae_test_feature_vector)
			# Calculate autoencoder accuracy
			ae_accuracy = accuracy_score(test_labels, ae_predicted_labels)
			ae_accuracies.append(ae_accuracy)
		
			print(f"p = {p}: AE Accuracy = {ae_accuracy:.2%}")

			if(plot_umaps):
				plot_umap(pca_train_feature_vector, train_labels, pca_test_feature_vector, test_labels, pca_predicted_labels, "PCA", p)
				plot_umap(lda_train_feature_vector, train_labels, lda_test_feature_vector, test_labels, lda_predicted_labels, "LDA", p)
				plot_umap(ae_train_feature_vector, ae_train_labels, ae_test_feature_vector, ae_test_labels, ae_predicted_labels, "AE", p)

	plt.plot(p_set, pca_accuracies, "-o", label="PCA")
	plt.plot(p_set, lda_accuracies, "-x", label="LDA")
	plt.plot(ae_p_set, ae_accuracies, "-*", label="AE")
	plt.title("Accuracy Comparison")
	plt.xlabel("Number of Eigenvectors P")
	plt.ylabel("Accuracy (%)")
	plt.legend()
	plt.show()




