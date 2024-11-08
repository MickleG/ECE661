import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

def compute_histogram(img):
	histogram = np.zeros(256, dtype=int)
	for pixel in img.ravel():
		histogram[int(pixel)] += 1

	return histogram

def find_otsu_threshold(histogram):

	max_variance = -np.inf
	w_0 = 0
	w_1 = 0
	sum_0 = 0
	sum_1 = 0

	threshold = 0

	total_sum = sum(histogram * np.arange(256))
	
	# Using 256 bins due to 0-255 being standard grayscale range
	for k in range(256):

		w_0 = sum(histogram[:k])
		w_1 = sum(histogram[k+1:])
		if(w_0 == 0 or w_1 == 0):
			continue

		sum_0 += (k * histogram[k])
		sum_1 = int(total_sum - sum_0)

		between_class_variance = w_0 * w_1 * (sum_0 / w_0 - sum_1 / w_1) ** 2

		if(between_class_variance > max_variance):
			max_variance = between_class_variance
			threshold = k


	return threshold



def otsu_rgb(img, max_iterations, namestring):
	mask = np.ones(img.shape[:2], dtype=np.uint8)

	title_list = ["red", "green", "blue"]

	cv2.imshow("channel blue", img[:, :, 0] * mask)
	cv2.imshow("channel green", img[:, :, 1] * mask)
	cv2.imshow("channel red", img[:, :, 2] * mask)
	
	cv2.waitKey(0)

	for iteration in range(max_iterations):
		channel_thresholds = []

		for i in range(3):
			channel = img[:, :, i] * mask

			histogram = compute_histogram(channel)

			threshold = find_otsu_threshold(histogram)

			channel_mask = np.where(channel >= threshold, 1, 0).astype(np.uint8)
			# mask[channel < threshold] = 0

			if(save_images and iteration == max_iterations - 1):
				save_channel_mask = channel_mask * 255
				cv2.imwrite('pics/' + namestring + '_rgb_mask_' + title_list[i] + '.jpg', save_channel_mask, [cv2.IMWRITE_JPEG_QUALITY, 90])

			channel_thresholds.append(channel_mask)

		# combining channel thresholds with boolean AND
		combined_mask = channel_thresholds[0]
		for i in range(len(channel_thresholds)):
			combined_mask = cv2.bitwise_and(combined_mask, channel_thresholds[i])

		mask = combined_mask.copy()


	return mask * 255

def otsu_texture(img, window_list, max_iterations, namestring):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask = np.ones(img.shape[:2], dtype=np.uint8)


	for iteration in range(max_iterations):
		channel_thresholds = []
		channels = []

		gray = gray * mask

		for window_size in window_list:
			pad_size = window_size // 2
			padded_img = np.pad(gray, pad_size, mode='constant', constant_values=0)
			variance_img = np.zeros(gray.shape)

			for v in range(img.shape[0]):
				for u in range(img.shape[1]):
					min_u = u - pad_size
					max_u = u + pad_size + 1
					min_v = v - pad_size
					max_v = v + pad_size + 1

					window = padded_img[min_v:max_v, min_u:max_u]

					if window.size > 0:
						mean_intensity = np.mean(window)
						variance_img[v, u] = np.var(window - mean_intensity)
					else:
						variance_img[v, u] = 0

			variance_img = cv2.normalize(variance_img, None, 0, 255, norm_type=cv2.NORM_MINMAX)
			# variance_img = (variance_img - np.min(variance_img)) / (np.max(variance_img) - np.min(variance_img) + 1e-6) * 255

			channels.append(variance_img)

		for i in range(3):
			channel = channels[i] * mask

			histogram = compute_histogram(channel)

			threshold = find_otsu_threshold(histogram)

			channel_mask = np.where(channel >= threshold, 1, 0).astype(np.uint8)

			if(save_images and iteration == max_iterations - 1):
				save_channel_mask = channel_mask * 255
				cv2.imwrite('pics/' + namestring + '_texture_mask_' + str(window_list[i]) + '.jpg', save_channel_mask, [cv2.IMWRITE_JPEG_QUALITY, 90])

			channel_thresholds.append(channel_mask)

		# combining channel thresholds with boolean AND
		combined_mask = channel_thresholds[0]
		for i in range(len(channel_thresholds)):
			combined_mask = cv2.bitwise_and(combined_mask, channel_thresholds[i])

		mask = combined_mask.copy()

	return mask * 255

def find_contours(mask, namestring):
	kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
	subtraction_img = cv2.erode(mask, kernel)

	if(save_images):
		cv2.imwrite('pics/' + namestring + '_contours.jpg', mask - subtraction_img, [cv2.IMWRITE_JPEG_QUALITY, 90])


save_images = True

img_dog = cv2.imread('pics/dog_small.jpg')
img_flower = cv2.resize(cv2.imread('pics/flower_small.jpg'), (504, 672))
img_fox = cv2.imread('pics/fox.jpg')
img_whale = cv2.resize(cv2.imread('pics/whale.jpg'), (501, 333))

foreground_rgb_dog = otsu_rgb(img=img_dog, max_iterations=4, namestring="dog")
foreground_texture_dog = otsu_texture(img=img_dog, window_list=[11, 17, 23], max_iterations=1, namestring="dog")
if(save_images):
	cv2.imwrite('pics/dog_rgb_mask_combined.jpg', foreground_rgb_dog, [cv2.IMWRITE_JPEG_QUALITY, 90])
	cv2.imwrite('pics/dog_texture_mask_combined.jpg', foreground_texture_dog, [cv2.IMWRITE_JPEG_QUALITY, 90])

foreground_rgb_flower = otsu_rgb(img=img_flower, max_iterations=4, namestring="flower")
foreground_texture_flower = otsu_texture(img=img_flower, window_list=[5, 7, 9], max_iterations=1, namestring="flower")
if(save_images):
	cv2.imwrite('pics/flower_rgb_mask_combined.jpg', foreground_rgb_flower, [cv2.IMWRITE_JPEG_QUALITY, 90])
	cv2.imwrite('pics/flower_texture_mask_combined.jpg', foreground_texture_flower, [cv2.IMWRITE_JPEG_QUALITY, 90])

foreground_rgb_fox = otsu_rgb(img=img_fox, max_iterations=4, namestring="fox")
foreground_texture_fox = otsu_texture(img=img_fox, window_list=[5, 7, 9], max_iterations=1, namestring="fox")
if(save_images):
	cv2.imwrite('pics/fox_rgb_mask_combined.jpg', foreground_rgb_fox, [cv2.IMWRITE_JPEG_QUALITY, 90])
	cv2.imwrite('pics/fox_texture_mask_combined.jpg', foreground_texture_fox, [cv2.IMWRITE_JPEG_QUALITY, 90])

foreground_rgb_whale = otsu_rgb(img=img_whale, max_iterations=4, namestring="whale")
foreground_texture_whale = otsu_texture(img=img_whale, window_list=[5, 7, 9], max_iterations=1, namestring="whale")
if(save_images):
	cv2.imwrite('pics/whale_rgb_mask_combined.jpg', foreground_rgb_whale, [cv2.IMWRITE_JPEG_QUALITY, 90])
	cv2.imwrite('pics/whale_texture_mask_combined.jpg', foreground_texture_whale, [cv2.IMWRITE_JPEG_QUALITY, 90])

find_contours(foreground_rgb_dog, "dog_rgb")
find_contours(foreground_texture_dog, "dog_texture")

find_contours(foreground_rgb_flower, "flower_rgb")
find_contours(foreground_texture_flower, "flower_texture")

find_contours(foreground_rgb_fox, "fox_rgb")
find_contours(foreground_texture_fox, "fox_texture")

find_contours(foreground_rgb_whale, "whale_rgb")
find_contours(foreground_texture_whale, "whale_texture")








