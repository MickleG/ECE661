import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_census_transform(image, window_size):
	rows, cols = image.shape
	half_window = window_size // 2
	census = np.zeros((rows, cols, window_size**2), dtype=np.uint8)

	for i in range(half_window, rows - half_window):
		for j in range(half_window, cols - half_window):
			center_pixel = image[i, j]
			bit_vector = []

			for m in range(-half_window, half_window + 1):
				for n in range(-half_window, half_window + 1):
					neighbor_pixel = image[i + m, j + n]
					if neighbor_pixel > center_pixel:
						bit_vector.append(1)
					else:
						bit_vector.append(0)

			census[i, j] = bit_vector

	return census

def census_disparity_map(image1, image2, window_size, dmax):
	rows, cols = image1.shape

	disparity_map = np.zeros((rows, cols), dtype=np.uint8)
	half_window = window_size // 2

	census1 = compute_census_transform(image1, window_size)
	census2 = compute_census_transform(image2, window_size)

	for i in range(half_window, rows - half_window):
		for j in range(half_window, cols - half_window):
			min_cost = float("inf")
			best_disparity = 0

			for d in range(dmax + 1):
				if j - d < half_window:
					# Skipping disparities that go out of bounds
					continue

				bitvector1 = census1[i, j]
				bitvector2 = census2[i, j - d]

				xor = np.bitwise_xor(bitvector1, bitvector2)
				data_cost = np.sum(xor)

				if data_cost < min_cost:
					min_cost = data_cost
					best_disparity = d

			disparity_map[i, j] = best_disparity

	return disparity_map


if __name__ == "__main__":
	image1 = cv2.imread("Task3Images/im2.png", cv2.IMREAD_GRAYSCALE)
	image2 = cv2.imread("Task3Images/im6.png", cv2.IMREAD_GRAYSCALE)

	ground_truth = cv2.imread("Task3Images/disp2.png", cv2.IMREAD_GRAYSCALE)
	ground_truth = np.array(np.array(ground_truth, dtype=np.float32) / 4, dtype=np.uint8)

	window_size = 11
	dmax = 64
	disparity_threshold = 2
	num_valid_points = 0
	num_accurate_points = 0

	disparity_map = census_disparity_map(image1, image2, window_size, dmax)
	error_mask = np.zeros_like(disparity_map)

	for i in range(disparity_map.shape[0]):
		for j in range(disparity_map.shape[1]):
			ground_truth_value = ground_truth[i, j]

			if ground_truth_value > 0:
				num_valid_points += 1

			if(ground_truth_value == 0):
				continue

			if(abs(ground_truth_value - disparity_map[i, j]) <= disparity_threshold):
				error_mask[i, j] = 255
				num_accurate_points += 1


	print("accuracy: ", num_accurate_points / num_valid_points)
	
	cv2.imshow("Disparity Map", disparity_map)
	cv2.imshow("Ground Truth", ground_truth)
	cv2.imshow("Error Mask", error_mask)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# 5x5 40% accuracy
# 11x11 51% accuracy


