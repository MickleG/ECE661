import cv2
import numpy as np
import time
import random
from sklearn.cluster import DBSCAN

# Canny edge detection parameters
canny_threshold1 = 200
canny_threshold2 = 400

# Hough transform parameters
rho_threshold = 1
theta_threshold = np.pi / 180
hough_threshold = 50

distance_threshold = 30

def HC(point):
	return np.array([point[0], point[1], 1])

def polar_to_cartesian(rho, theta, length=1500):
	x0 = rho * np.cos(theta)
	y0 = rho * np.sin(theta)
	x1 = int(x0 + length * (-np.sin(theta)))
	y1 = int(y0 + length * (np.cos(theta)))
	x2 = int(x0 - length * (-np.sin(theta)))
	y2 = int(y0 - length * (np.cos(theta)))

	return (x1, y1, x2, y2)

def detect_corners(image, visualize=True):

	# Performing canny edge detection
	edges = cv2.Canny(image, canny_threshold1, canny_threshold2)

	detected_edge_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

	image_height, image_width = image.shape

	# Performing hough transform to extract lines
	lines = cv2.HoughLines(edges, rho_threshold, theta_threshold, hough_threshold)

	# Filtering out duplicate lines for the same edge that deviate by small angles
	rho_filter = 15
	theta_filter = np.deg2rad(5)
	print("theta filter: ", theta_filter)

	filtered_lines = []

	print("first line: ", lines[0])

	# Sorting lines by rho
	if lines is not None:
		lines = sorted(lines, key=lambda x: x[0][0])

	# Checking if both rho and theta differences are large to qualify as a unique line
	for line in lines:
		# collect individual line's rho and theta values
		rho, theta = line[0]
		is_unique = True

		for unique_rho, unique_theta, _ in filtered_lines:
			# Checking if rho falls within a threshold of any previous rho as well as theta falling within a theta threshold of any previous theta to check if lines are duplicates. Check is complicated due to bounds of rho and theta for Hough
			# First check is for positive rho, and therefor angle differences close to zero. If rho and unique_rho are opposite in sign, angle will be close to pi if lines are duplicates
			if (abs(rho - unique_rho) < rho_filter and abs(theta - unique_theta) < theta_filter) or (abs(abs(rho) - abs(unique_rho)) < rho_filter and abs(theta - unique_theta) > (np.pi - theta_filter)):
				is_unique = False
				break
		
		# If unique flag still true, line passed checking against all other lines and is in fact not a duplicate
		if is_unique:
			x1, y1, x2, y2 = polar_to_cartesian(rho, theta)
			line_cartesian = (x1, y1, x2, y2)

			filtered_lines.append((rho, theta, line_cartesian))

	

	# Sorting lines by rho and vertical/horizontal to provide consistent labelling for each eventually found corner
	vertical_lines = []
	horizontal_lines = []

	for line in filtered_lines:
		rho, theta, line_cartesian = line
		if(abs(theta) < np.deg2rad(50) or abs(theta - np.pi) < np.deg2rad(50)):
			vertical_lines.append(line)
		elif(abs(theta - np.pi/2) < np.deg2rad(50) or abs(theta - 3*np.pi/2) < np.deg2rad(50)):
			horizontal_lines.append(line)

	# Sorting by abs for vertical lines only as they were more prone to angle switching
	vertical_lines.sort(key=lambda x: abs(x[0]))
	horizontal_lines.sort(key=lambda x: x[0])

	# for line in horizontal_lines:
	# 	color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
	# 	x1, y1, x2, y2 = line[2]
	# 	rho = line[0]

	# 	midpoint = (int((x1 + x2) / 2) + 500, int((y1 + y2) / 2) + 400)
	# 	cv2.line(detected_edge_image, (x1, y1), (x2, y2), color, 2)
	# 	cv2.putText(detected_edge_image, "rho: {}".format(rho), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


	filtered_lines = horizontal_lines + vertical_lines


	# Finding corners using cross products in HC
	corners = []

	for i in range(len(filtered_lines)):
		line_cartesian_test = filtered_lines[i][2]

		endpoint1_hc_test = HC((line_cartesian_test[0], line_cartesian_test[1]))
		endpoint2_hc_test = HC((line_cartesian_test[2], line_cartesian_test[3]))

		line_hc_test = np.cross(endpoint1_hc_test, endpoint2_hc_test)

		for j in range(len(filtered_lines)):
			if(i != j):
				line_cartesian = filtered_lines[j][2]

				endpoint1_hc = HC((line_cartesian[0], line_cartesian[1]))
				endpoint2_hc = HC((line_cartesian[2], line_cartesian[3]))

				line_hc = np.cross(endpoint1_hc, endpoint2_hc)

				intersection = np.cross(line_hc_test, line_hc)

				is_valid_corner = True
				is_unique_corner = True

				if(intersection[2] != 0):
					intersection = intersection / intersection[2]
					if(intersection[0] < 0 or intersection[1] < 0 or intersection[1] > image_height or intersection[0] > image_width):
						is_valid_corner = False # Intersection out of the bounds of the image
				else:
					is_valid_corner = False # Intersection is ideal point

				
				for corner in corners:
					if(intersection[0] == corner[0] and intersection[1] == corner[1]):
						is_unique_corner = False
						break

				if is_unique_corner and is_valid_corner:		
					corners.append(intersection)

	print("num corners: ", len(corners))


	# Visualizing detected lines
	if(visualize):
		for i, filtered_line in enumerate(filtered_lines):
			x1, y1, x2, y2 = filtered_line[2]
			cv2.line(detected_edge_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

		for i, corner in enumerate(corners):
			corner_location = (int(corner[0]), int(corner[1]))
			cv2.circle(detected_edge_image, corner_location, 2, (0, 0, 255), -1)
			cv2.putText(detected_edge_image, "{}".format(i), corner_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

		cv2.imshow("original", image)
		cv2.imshow("edges", edges)
		cv2.imshow("hough after filtering", detected_edge_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()




for i in range(1, 41):
	image = cv2.imread('HW8-Files/Dataset1/Pic_' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)

	detect_corners(image)




