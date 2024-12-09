import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# Define paths to the training and test data
train_dir = "CarDetection/train"
test_dir = "CarDetection/test"
positive_train_dir = os.path.join(train_dir, "positive")
negative_train_dir = os.path.join(train_dir, "negative")
positive_test_dir = os.path.join(test_dir, "positive")
negative_test_dir = os.path.join(test_dir, "negative")

num_iterations = 20

class WeakClassifier:
	def __init__(self, feature, threshold, polarity):
		"""
		feature: the Haar-like feature on which the classifier is based
		threshold: the threshold value used for the decision
		polarity: the direction of the decision (whether the classifier should classify 
				  as positive or negative when the feature is greater or smaller than the threshold)
		"""
		self.feature = feature
		self.threshold = threshold
		self.polarity = polarity

	def predict(self, features):
		"""
		Predict whether the image region satisfies the condition of this weak classifier.
		"""
		feature_value = features[self.feature]

		if self.polarity == 1:
			return 1 if feature_value >= self.threshold else -1
		else:
			return 1 if feature_value < self.threshold else -1

class AdaBoost:
	def __init__(self, T, max_stages=5):
		"""
		T: the number of rounds (weak classifiers to be trained)
		"""
		self.T = T
		self.alphas = []
		self.max_stages = max_stages
		self.classifiers = []

	def fit(self, X, y):
		#Fit the AdaBoost model to the training data.
		#X: features (integral image)
		#y: labels (1 or -1 for each image)

		# Initial weight for each sample
		w = np.ones(len(X)) / len(X)
		
		for t in tqdm(range(self.T), desc="Training AdaBoost", ncols=100):
			if self.max_stages is not None and len(self.classifiers) >= self.max_stages:
				break

			# Select the weak classifier with the minimum weighted error
			best_classifier = None
			min_error = float('inf')
			best_threshold = 0
			best_polarity = 1
			
			# Iterate through the features to select the best weak classifier
			for feature_index in range(len(X[0])):
				feature_values = [features[feature_index] for features in X]
				sorted_indices = np.argsort(feature_values)
				sorted_features = np.array(feature_values)[sorted_indices]
				sorted_weights = w[sorted_indices]
				sorted_labels = np.array(y)[sorted_indices]

				T_plus = np.sum(w * (y == 1))
				T_minus = np.sum(w * (y == -11))

				for i in range(1, len(X)):
					threshold = (sorted_features[i - 1] + sorted_features[i]) / 2
					S_plus = np.sum(sorted_weights[:i] * (sorted_labels[:i] == 1))
					S_minus = np.sum(sorted_weights[:i] * (sorted_labels[:i] == -11))

					error_pos_1 = S_plus + (T_minus - S_minus)
					error_neg_1 = S_minus + (T_plus - S_plus)

					error = min(error_pos_1, error_neg_1)

					if error < min_error:
						min_error = error
						best_classifier = WeakClassifier(feature_index, threshold, 1)
						best_threshold = threshold
						best_polarity = 1

			if best_classifier is None:
				print("No valid weak classifier found in this round")
				break

			# Compute alpha for the weak classifier
			alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))
			self.alphas.append(alpha)
			self.classifiers.append(best_classifier)

			# Update weights based on the classifier's performance
			predictions = np.array([best_classifier.predict(features) for features in X])
			w = w * np.exp(-alpha * y * predictions)
			w = w / np.sum(w)  # Normalize weights to sum to 1

	def predict(self, X):
		# Predict the labels for the test data.
		strong_preds = np.zeros(len(X))
		
		for alpha, classifier in zip(self.alphas, self.classifiers):
			predictions = np.array([classifier.predict(features) for features in X])
			strong_preds += alpha * predictions
		
		return np.sign(strong_preds)

class CascadeClassifier:
	def __init__(self, thresholds, max_stages=5):
		"""
		thresholds: a list of thresholds for each stage of the cascade.
		"""
		self.stages = []
		self.thresholds = thresholds
		self.fp_rates = []
		self.fn_rates = []
		self.max_stages = max_stages

	def train(self, X, y):
		"""
		Train the cascade classifier.
		X: training data (integral images)
		y: labels (1 or -1)
		"""
		stage_data = np.array(X)
		y = np.array(y)

		# print("thresholds: ", self.thresholds)

		for stage_index, threshold in enumerate(self.thresholds):
			if self.max_stages is not None and len(self.stages) >= self.max_stages:
				break

			# print("threshold: ", threshold)
			adaboost = AdaBoost(T=num_iterations)
			adaboost.fit(stage_data, y)
			self.stages.append(adaboost)

			# Get predictions on the current stage's data
			predictions = adaboost.predict(stage_data)

			# Compute False Positive and False Negative rates
			fp = np.sum((predictions == 1) & (y == 0))  # Positive classified as negative
			fn = np.sum((predictions == -1) & (y == 1))  # Negative classified as positive

			fp_rate = fp / np.sum(y == 0) if np.sum(y == 0) > 0 else 0
			fn_rate = fn / np.sum(y == 1) if np.sum(y == 1) > 0 else 0

			# Store FP and FN rates for plotting
			self.fp_rates.append(fp_rate)
			self.fn_rates.append(fn_rate)

			print("fp_rate: ", fp_rate)
			print("fn_rate: ", fn_rate)

			# Keep only the images that passed the current stage
			passed_indices = (predictions == 1)
			stage_data = stage_data[passed_indices]
			y = y[passed_indices]

			# Optionally, break early if no images remain
			if len(stage_data) == 0:
				break

		# Plot the FP and FN rates
		self.plot_fp_fn()

	def plot_fp_fn(self):
		"""
		Plot the FP and FN rates as a function of the number of stages.
		"""
		plt.figure(figsize=(10, 5))
		plt.plot(range(1, len(self.fp_rates) + 1), self.fp_rates, label="False Positive Rate", color='r')
		plt.plot(range(1, len(self.fn_rates) + 1), self.fn_rates, label="False Negative Rate", color='b')
		plt.xlabel("Number of Stages")
		plt.ylabel("Rate")
		plt.title("FP and FN Rates as a Function of Cascade Stages")
		plt.legend()
		plt.grid(True)
		plt.show()

	def predict(self, X):
		# Apply the cascade to test data.
		for stage in self.stages:
			predictions = stage.predict(X)
			# If any stage fails to detect the object, reject the image
			if np.any(predictions == -1):
				return -1
		return 1


# Function to load images and labels
def load_data(positive_dir, negative_dir):
	images = []
	labels = []
	
	# Load positive images
	for filename in os.listdir(positive_dir):
		img = cv2.imread(os.path.join(positive_dir, filename), cv2.IMREAD_GRAYSCALE)
		images.append(img)
		labels.append(1)  # Positive label
	
	# Load negative images
	for filename in os.listdir(negative_dir):
		img = cv2.imread(os.path.join(negative_dir, filename), cv2.IMREAD_GRAYSCALE)
		images.append(img)
		labels.append(0)  # Negative label
	
	return images, labels


def compute_integral_image(image):
	return cv2.integral(image)[1:, 1:]

def extract_haar_features(integral_image, img_width, img_height):
	"""
	Extracts basic Haar-like features from the integral image.
	"""
	features = []

	# Horizontal 1x4 filter
	for y in range(img_height):
		for x in range(img_width - 4):  # Ensure filter fits within width
			left = integral_image[y, x]
			middle_left = integral_image[y, x + 1]
			middle_right = integral_image[y, x + 2]
			right = integral_image[y, x + 3]
			
			horizontal_feature = (middle_right + right) - (middle_left + left)
			features.append(horizontal_feature)

	# Vertical 4x1 filter
	for y in range(img_height - 4):  # Ensure filter fits within height
		for x in range(img_width):
			top = integral_image[y, x]
			middle_top = integral_image[y + 1, x]
			middle_bottom = integral_image[y + 2, x]
			bottom = integral_image[y + 3, x]
			
			vertical_feature = (middle_bottom + bottom) - (middle_top + top)
			features.append(vertical_feature)

	return np.array(features)

def evaluate_accuracy(model, X_test, y_test):
	"""
	Function to evaluate accuracy of a model on test data.
	"""
	predictions = model.predict(X_test)
	accuracy = np.mean(predictions == y_test)
	return accuracy


if __name__ == "__main__":
	# Load the training data
	train_images, train_labels = load_data(positive_train_dir, negative_train_dir)
	test_images, test_labels = load_data(positive_test_dir, negative_test_dir)

	X_train = [extract_haar_features(compute_integral_image(image), img_width=image.shape[1], img_height=image.shape[0]) for image in train_images]
	y_train = np.array(train_labels)

	cascade = CascadeClassifier(thresholds=[0.5, 0.4, 0.3])
	cascade.train(X_train, y_train)

	X_test = [extract_haar_features(compute_integral_image(image), img_width=image.shape[1], img_height=image.shape[0]) for image in test_images]
	y_test = np.array(test_labels)

	accuracy = evaluate_accuracy(cascade, X_test, y_test)
	print(f"Accuracy: {accuracy * 100:.2f}%")















