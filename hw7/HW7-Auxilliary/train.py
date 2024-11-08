import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

train_features = np.random.rand(922, 10)
train_labels = np.random.randint(0, 4, size=(922,))
test_features = np.random.rand(200, 10)
test_labels = np.random.randint(0, 4, size=(200,))

scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

parameter_grid = {
	'C': [0.1, 1, 10, 100],
	'kernel': ['linear', 'rbf'],
	'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(svm.SVC(), parameter_grid, cv=5)
grid_search.fit(train_features, train_labels)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(test_features)

print("Accuracy:", accuracy_score(test_labels, y_pred))
print("Classification Report:\n", classification_report(test_labels, y_pred, zero_division=0))


conf_matrix = confusion_matrix(test_labels, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(test_labels))
disp.plot(cmap='Blues')

plt.show()
# def train_svm_classifier(train_features, train_labels, test_features, test_labels, batch_size):
# 	# Create SVM classifier
# 	clf = svm.SVC(kernel='sigmoid')  # You can change the kernel as needed

# 	# Train the model in batches
# 	for i in range(0, len(train_features), batch_size):
# 		X_batch = train_features[i:i + batch_size]
# 		y_batch = train_labels[i:i + batch_size]
# 		if len(np.unique(y_batch)) > 1:
# 			clf.fit(X_batch, y_batch)
# 		else:
# 			print(f"Skipping batch {i // batch_size} due to only one unique class.")

# 	# Make predictions on the test set
# 	y_pred = clf.predict(test_features)

# 	# Evaluate the model
# 	print("Accuracy:", accuracy_score(test_labels, y_pred))
# 	print("Classification Report:\n", classification_report(test_labels, y_pred, zero_division=0))

# # Example of training with a specified batch size
# batch_size = 8  # Change this value as needed
# train_svm_classifier(train_features, train_labels, test_features, test_labels, batch_size)