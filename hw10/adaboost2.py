import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

num_iterations = 20

class WeakClassifier:
	def __init__(self, feature, threshold, polarity):
		self.feature = feature
		self.threshold = threshold
		self.polarity = polarity

	def predict(self, features):
		feature_value = features[self.feature]
		return 1 if (self.polarity == 1 and feature_value >= self.threshold) or (self.polarity == -1 and feature_value < self.threshold) else -1


class AdaBoost:
	def __init__(self, T):
		self.T = T
		self.alphas = []
		self.classifiers = []

	def fit(self, X, y):
		w = np.ones(len(X)) / len(X)  # Initial weight for each sample
		
		for t in tqdm(range(self.T), desc="Training AdaBoost", ncols=100):
			best_classifier = None
			min_error = float('inf')
			
			for feature_index in range(len(X[0])):
				# Sorting features
				feature_values = [features[feature_index] for features in X]
				sorted_indices = np.argsort(feature_values)
				sorted_features = np.array(feature_values)[sorted_indices]
				sorted_weights = w[sorted_indices]
				sorted_labels = np.array(y)[sorted_indices]
				
				# Calculating next threshold value
				T_plus = np.sum(w * (y == 1))
				T_minus = np.sum(w * (y == -1))

				for i in range(1, len(X)):
					threshold = (sorted_features[i - 1] + sorted_features[i]) / 2
					S_plus = np.sum(sorted_weights[:i] * (sorted_labels[:i] == 1))
					S_minus = np.sum(sorted_weights[:i] * (sorted_labels[:i] == -1))

					error_pos_1 = S_plus + (T_minus - S_minus)
					error_neg_1 = S_minus + (T_plus - S_plus)

					# Choosing minimum between two error metrics
					error = min(error_pos_1, error_neg_1)

					if error < min_error:
						min_error = error
						best_classifier = WeakClassifier(feature_index, threshold, 1)

			# Find new parameters to compute next weak classifier
			alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))
			self.alphas.append(alpha)
			self.classifiers.append(best_classifier)

			predictions = np.array([best_classifier.predict(features) for features in X])
			w = w * np.exp(-alpha * y * predictions)
			w = w / np.sum(w)  # Normalize weights


	def predict(self, X):
		strong_preds = np.zeros(len(X))
		for alpha, classifier in zip(self.alphas, self.classifiers):
			predictions = np.array([classifier.predict(features) for features in X])
			strong_preds += alpha * predictions
		return np.sign(strong_preds)


class CascadeClassifier:
	def __init__(self, false_positive_target, true_detection_target, num_stages):
		self.false_positive_target = false_positive_target
		self.true_detection_target = true_detection_target
		self.num_stages = num_stages
		self.stages = []
		self.fp_rates = []
		self.fn_rates = []

	def train(self, X, y):

		for stage_index in range(self.num_stages):
			print(f"Training stage {stage_index + 1}/{self.num_stages}...")
			# Perform adaboost to fit weak classifier to data
			adaboost = AdaBoost(T=num_iterations)
			adaboost.fit(X, y)
			self.stages.append(adaboost)

			# Get predictions on the current stage's data
			predictions = adaboost.predict(X)

			# Compute False Positive and False Negative rates
			fp = np.sum((predictions == 1) & (y == 0))  # Positive classified as negative
			fn = np.sum((predictions == -1) & (y == 1))  # Negative classified as positive

			fp_rate = fp / np.sum(y == 0) if np.sum(y == 0) > 0 else 0
			fn_rate = fn / np.sum(y == 1) if np.sum(y == 1) > 0 else 0

			self.fp_rates.append(fp_rate)
			self.fn_rates.append(fn_rate)

			print(f"Stage {stage_index + 1}: FP rate = {fp_rate}, FN rate = {fn_rate}")

			# If the stage does not meet target FP and FN rates, stop training
			if fp_rate > self.false_positive_target or fn_rate < self.true_detection_target:
				print(f"Stage {stage_index + 1} did not meet target rates. Stopping early.")
				break

			# Keep only the samples that pass the current stage
			passed_indices = (predictions == 1)
			X = np.array(X)[passed_indices.astype(int)]
			y = np.array(y)[passed_indices.astype(int)]

			if len(X) == 0:
				break

		self.plot_performance()

	def predict(self, X):
		for stage in self.stages:
			predictions = stage.predict(X)
			if np.any(predictions == -1):  # Reject if any stage fails
				return -1
		return 1

	def plot_performance(self):
		# Plotting FP and FN rates
		stages = np.arange(1, len(self.fp_rates) + 1)

		plt.figure(figsize=(10, 6))

		# Plot FP rate
		plt.plot(stages, self.fp_rates, label="FP Rate", color="red", marker='o')
		# Plot FN rate
		plt.plot(stages, self.fn_rates, label="FN Rate", color="blue", marker='x')

		# Labels and title
		plt.xlabel("Cascade Stage")
		plt.ylabel("Rate")
		plt.title("False Positive Rate and False Negative Rate as a Function of Cascade Stages")

		# Show a legend
		plt.legend()

		# Show the plot
		plt.show()


def extract_haar_features(integral_image):
	img_height, img_width = integral_image.shape
	features = []

	# Horizontal 1x2 feature
	for y in range(img_height):
		for x in range(img_width - 1):
			left = integral_image[y, x]
			right = integral_image[y, x + 1]
			horizontal_feature = right - left
			features.append(horizontal_feature)

	# Vertical 2x1 feature
	for y in range(img_height - 1):
		for x in range(img_width):
			top = integral_image[y, x]
			bottom = integral_image[y + 1, x]
			vertical_feature = bottom - top
			features.append(vertical_feature)

	return np.array(features)


def compute_integral_image(image):
	return image.cumsum(axis=0).cumsum(axis=1)


def load_data(positive_dir, negative_dir):
	images = []
	labels = []
	
	# Loading data and labels
	for filename in os.listdir(positive_dir):
		img = cv2.imread(os.path.join(positive_dir, filename), cv2.IMREAD_GRAYSCALE)
		images.append(img)
		labels.append(1)

	for filename in os.listdir(negative_dir):
		img = cv2.imread(os.path.join(negative_dir, filename), cv2.IMREAD_GRAYSCALE)
		images.append(img)
		labels.append(0)

	images = np.array(images)
	labels = np.array(labels)

	return images, labels


# Load training data
train_dir = "CarDetection/train"
positive_train_dir = os.path.join(train_dir, "positive")
negative_train_dir = os.path.join(train_dir, "negative")

positive_train_images, positive_train_labels = load_data(positive_train_dir, negative_train_dir)

train_integral_images = [compute_integral_image(img) for img in positive_train_images]
train_features = [extract_haar_features(integral_image) for integral_image in train_integral_images]

# Train cascade classifier
cascade_classifier = CascadeClassifier(false_positive_target=0.1, true_detection_target=0.9, num_stages=5)
cascade_classifier.train(train_features, positive_train_labels)

# Test the classifier
test_dir = "CarDetection/test"
positive_test_dir = os.path.join(test_dir, "positive")
negative_test_dir = os.path.join(test_dir, "negative")

positive_test_images, positive_test_labels = load_data(positive_test_dir, negative_test_dir)

test_integral_images = [compute_integral_image(img) for img in positive_test_images]
test_features = [extract_haar_features(integral_image) for integral_image in test_integral_images]

test_predictions = cascade_classifier.predict(test_features)
# test_predictions = [cascade_classifier.predict(features) for features in test_features]
accuracy = np.mean(np.array(test_predictions) == np.array(positive_test_labels))
print(f"Cascade Classifier Test Accuracy: {accuracy * 100:.2f}%")




