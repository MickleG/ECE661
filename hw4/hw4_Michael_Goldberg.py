import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def harris_corner_detection(image, sigma, k, namestring):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255

	# Finding M (lowest even integer > 4*sigma)
	if(int(sigma * 4) % 2 == 0):
		kernel_size = int(sigma * 4) + 2
	else:
		kernel_size = int(sigma * 4) + 1

	# Creating and applying haar wavelet kernels
	kernel_x = np.ones((kernel_size, kernel_size), dtype=np.float32)
	kernel_y = np.ones((kernel_size, kernel_size), dtype=np.float32)
	kernel_x[:, :int(kernel_size / 2)] = -1
	kernel_y[int(kernel_size / 2):, :] = -1
	dx = cv2.filter2D(gray, -1, kernel_x)
	dy = cv2.filter2D(gray, -1, kernel_y)


	# Computing C matrix components
	dx2 = dx ** 2
	dy2 = dy ** 2
	dxdy = dx * dy

	# Kernel of 1s to perform a sum across the kernel
	summation_kernel = np.ones((int(5 * sigma), int(5 * sigma)), dtype=np.float32)
	sum_dx2 = cv2.filter2D(dx2, -1, summation_kernel)
	sum_dy2 = cv2.filter2D(dy2, -1, summation_kernel)
	sum_dxdy = cv2.filter2D(dxdy, -1, summation_kernel)

	# Finding harris response and thresholding
	detC = (sum_dx2 * sum_dy2) - (sum_dxdy ** 2)
	traceC = sum_dx2 + sum_dy2

	response = detC - k * (traceC ** 2)
	response_threshold = np.mean(np.abs(response))


	corner_image = np.copy(image)


	# Reducing number of points to most prominent 100
	corners = np.empty((0, 3))
	corner_image = np.copy(image)
	N = int(10 * sigma)

	for x in range(N, gray.shape[1] - N):
		for y in range(N, gray.shape[0] - N):
			response_window = response[y - N:y + N + 1, x - N:x + N + 1]
			response_max = np.max(response_window)
			if response[y, x] == response_max and response_max > response_threshold:
				corners = np.vstack((corners, np.array([x, y, response[y, x]])))

	corners_filtered = np.array(corners[np.argsort(corners[:, 2])])[-100:,:2].astype(int)

	for corner in corners_filtered:
			cv2.circle(corner_image, (corner[0], corner[1]), radius = 4, color = (0, 0, 255), thickness = -1)

	# plt.imshow(corner_image)
	# plt.show()

	cv2.imwrite('HW4_images/harris_' + namestring + '_' + str(sigma) + '.jpg', corner_image, [cv2.IMWRITE_JPEG_QUALITY, 90])


	return corners_filtered


def SSD(img1, img2, img1_corners, img2_corners, kernel_size, namestring):
	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255

	# Drawing interest points first
	for corner in img1_corners:
		cv2.circle(img1, (corner[0], corner[1]), radius = 2, color = (0, 0, 255), thickness = -1)
	for corner in img2_corners:
		cv2.circle(img2, (corner[0], corner[1]), radius = 2, color = (0, 0, 255), thickness = -1)

	# Creating combined image for visualization
	combined_image = np.concatenate((img1, img2), axis = 1)

	half_k = kernel_size // 2
	matches = []

	# Converting img2 corners to list to allow for point removal to ensure no dual matches
	img2_corners = [tuple(corner) for corner in img2_corners]


	for (x1, y1) in img1_corners:
		best_match = None
		lowest_ssd = float('inf')

		# Skipping corners in img1 too close to edge
		if(x1 - half_k < 0 or x1 + half_k >= img1.shape[1] or y1 - half_k < 0 or y1 + half_k >= img1.shape[0]):
			continue

		# Creaitng kernel at each image1 feature point
		window1 = img1[y1 - half_k : y1 + half_k + 1, x1 - half_k : x1 + half_k + 1]

		for (x2, y2) in img2_corners:
			# Skipping corners in img2 too close to edge
			if(x2 - half_k < 0 or x2 + half_k >= img2.shape[1] or y2 - half_k < 0 or y2 + half_k >= img2.shape[0]):
				continue

			# Creating kernel at each image2 feature point
			window2 = img2[y2 - half_k : y2 + half_k + 1, x2 - half_k : x2 + half_k + 1]

			# Calculating SSD metric across both kernels
			ssd = np.sum((window1 - window2) ** 2)

			# Updating lowest ssd value
			if(ssd < lowest_ssd):
				lowest_ssd = ssd
				best_match = (x2, y2)

		# If a match is found, draw a line and remove img2 point from potential candidates for future image1 points
		if(best_match):
			cv2.line(combined_image, (x1, y1), (best_match[0] + img1.shape[1], best_match[1]), color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=1)
			img2_corners.remove(best_match)

	# plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
	# plt.show()

	cv2.imwrite('HW4_images/SSD_' + namestring + '.jpg', combined_image, [cv2.IMWRITE_JPEG_QUALITY, 90])


def NCC(img1, img2, img1_corners, img2_corners, kernel_size, namestring):
	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255

	for corner in img1_corners:
		cv2.circle(img1, (corner[0], corner[1]), radius = 2, color = (0, 0, 255), thickness = -1)
	for corner in img2_corners:
		cv2.circle(img2, (corner[0], corner[1]), radius = 2, color = (0, 0, 255), thickness = -1)

	combined_image = np.concatenate((img1, img2), axis = 1)

	half_k = kernel_size // 2
	matches = []

	img2_corners = [tuple(corner) for corner in img2_corners]

	for (x1, y1) in img1_corners:
		best_match = None
		highest_ncc = -1

		# Skipping corners in img1 too close to edge
		if(x1 - half_k < 0 or x1 + half_k >= img1.shape[1] or y1 - half_k < 0 or y1 + half_k >= img1.shape[0]):
			continue

		window1 = img1[y1 - half_k : y1 + half_k + 1, x1 - half_k : x1 + half_k + 1]

		window1_mean = np.mean(window1)

		for (x2, y2) in img2_corners:
			# Skipping corners in img2 too close to edge
			if(x2 - half_k < 0 or x2 + half_k >= img2.shape[1] or y2 - half_k < 0 or y2 + half_k >= img2.shape[0]):
				continue

			window2 = img2[y2 - half_k : y2 + half_k + 1, x2 - half_k : x2 + half_k + 1]

			window2_mean = np.mean(window2)

			# Calculating NCC metric piecewise
			numerator = np.sum((window1 - window1_mean) * (window2 - window2_mean))
			denominator = np.sqrt((np.sum(window1 - window1_mean)) ** 2 * np.sum((window2 - window2_mean)) ** 2)

			# Ensuring no division by zero
			if denominator != 0:
				ncc = numerator / denominator
			else:
				ncc = 0

			if(ncc > highest_ncc):
				highest_ncc = ncc
				best_match = (x2, y2)

		# If a match is found, draw a line and remove img2 point from potential candidates for future image1 points
		if(best_match):
			cv2.line(combined_image, (x1, y1), (best_match[0] + img1.shape[1], best_match[1]), color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=1)
			img2_corners.remove(best_match)

	# plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
	# plt.show()

	cv2.imwrite('HW4_images/NCC_' + namestring + '.jpg', combined_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

def sift(img1, img2, namestring):
	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	combined_image = np.concatenate((img1, img2), axis = 1)

	# Instantiating sift object, creating keypoints and descriptors
	sift = cv2.SIFT_create()
	keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
	keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

	# Using k-nearest neighbors for matching, and applying Lowe Ratio test to determine if match is strong
	matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

	matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

	good_matches = []
	for m, n in matches:
		if m.distance < 0.75 * n.distance:
			good_matches.append(m)

	# Draw lines between matches
	for match in good_matches:
		img1_idx = match.queryIdx
		img2_idx = match.trainIdx
		
		(x1, y1) = keypoints1[img1_idx].pt
		(x2, y2) = keypoints2[img2_idx].pt
		
		x2 += img1.shape[1]
		
		cv2.circle(combined_image, (int(x1), int(y1)), 2, color=(0, 0, 255), thickness=-1)
		cv2.circle(combined_image, (int(x2), int(y2)), 2, color=(0, 0, 255), thickness=-1)
		
		cv2.line(combined_image, (int(x1), int(y1)), (int(x2), int(y2)), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=1)

	cv2.imwrite('HW4_images/SIFT_' + namestring + '.jpg', combined_image, [cv2.IMWRITE_JPEG_QUALITY, 90])


hovde1 = cv2.imread('HW4_images/hovde_2.jpg')
hovde2 = cv2.imread('HW4_images/hovde_3.jpg')
temple1 = cv2.imread('HW4_images/temple_1.jpg')
temple2 = cv2.imread('HW4_images/temple_2.jpg')

keyboard1 = cv2.imread('HW4_images/keyboard1.jpg')
keyboard2 = cv2.imread('HW4_images/keyboard2.jpg')
oscilloscope1 = cv2.imread('HW4_images/oscilloscope1.jpg')
oscilloscope2 = cv2.imread('HW4_images/oscilloscope2.jpg')


sigmas = [0.8, 1.2, 1.4, 2] # replace with custom values

for sigma in sigmas:
	hovde1_corners = harris_corner_detection(hovde1, sigma, 0.05, 'hovde1')
	hovde2_corners = harris_corner_detection(hovde2, sigma, 0.05, 'hovde2')
	temple1_corners = harris_corner_detection(temple1, sigma, 0.05, 'temple1')
	temple2_corners = harris_corner_detection(temple2, sigma, 0.05, 'temple2')

	keyboard1_corners = harris_corner_detection(keyboard1, sigma, 0.05, 'keyboard1')
	keyboard2_corners = harris_corner_detection(keyboard2, sigma, 0.05, 'keyboard2')
	oscilloscope1_corners = harris_corner_detection(oscilloscope1, sigma, 0.05, 'oscilloscope1')
	oscilloscope2_corners = harris_corner_detection(oscilloscope2, sigma, 0.05, 'oscilloscope2')


SSD(hovde1, hovde2, hovde1_corners, hovde2_corners, 39, 'hovde')
SSD(temple1, temple2, temple1_corners, temple2_corners, 39, 'temple')
SSD(keyboard1, keyboard2, keyboard1_corners, keyboard2_corners, 21, 'keyboard')
SSD(oscilloscope1, oscilloscope2, oscilloscope1_corners, oscilloscope2_corners, 21, 'oscilloscope')

NCC(hovde1, hovde2, hovde1_corners, hovde2_corners, 39, 'hovde')
NCC(temple1, temple2, temple1_corners, temple2_corners, 39, 'temple')
NCC(keyboard1, keyboard2, keyboard1_corners, keyboard2_corners, 21, 'keyboard')
NCC(oscilloscope1, oscilloscope2, oscilloscope1_corners, oscilloscope2_corners, 21, 'oscilloscope')

sift(hovde1, hovde2, 'hovde')
sift(temple1, temple2, 'temple')
sift(keyboard1, keyboard2, 'keyboard')
sift(oscilloscope1, oscilloscope2, 'oscilloscope')