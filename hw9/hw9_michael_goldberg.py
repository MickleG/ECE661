import cv2
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from matplotlib.patches import ConnectionPatch
from scipy.optimize import least_squares
from scipy.linalg import null_space
from mpl_toolkits.mplot3d import Axes3D


image1 = cv2.imread("img1_alt.png")
image2 = cv2.imread("img2_alt.png")

image1_height, image1_width, _ = image1.shape
image2_height, image2_width, _ = image2.shape

# cv2.imshow("image1", image1)
# cv2.imshow("image2", image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

visualize = True

# Since we are using canonical form this is constant
P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])


def HC(point): # Helper function to turn pixels into HC
	return np.array([point[0], point[1], 1]).reshape((3, 1))

def dist(p1, p2):
	return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def condition_F(F):
	# Conditioning F by performing SVD and setting its smallest singular value in D to 0
	U, D, Vt = np.linalg.svd(F)
	D[-1] = 0

	D = np.diag(D)

	F_conditioned = np.dot(U, np.dot(D, Vt))

	return F_conditioned

def calculate_raster(image, H):
	height, width, _ = image.shape

	# Transforming corners through homography and choosing max bounds
	corners = np.array([
		[0, 0, 1],
		[width, 0, 1],
		[width, height, 1],
		[0, height, 1]
	]).T

	transformed_corners = H @ corners
	transformed_corners /= (transformed_corners[-1, :] + 1e-6)

	min_x = min(transformed_corners[0, :])
	max_x = max(transformed_corners[0, :])
	min_y = min(transformed_corners[1, :])
	max_y = max(transformed_corners[1, :])

	w_out = int(max_x - min_x)
	h_out = int(max_y - min_y)

	return w_out, h_out


def visualize_correspondences():

	for (x1, y1), (x2, y2) in zip(image1_points, image2_points):
		cv2.circle(combined_image, (x1, y1), radius=5, color=(0, 0, 255), thickness=-1)
		cv2.circle(combined_image, (x2 + image1_width, y2), radius=5, color=(0, 0, 255), thickness=-1)

		cv2.line(combined_image, (x1, y1), (x2 + image1_width, y2), color=(0, 255, 0), thickness=2)

	plt.imshow(combined_image)
	plt.show()

def normalize_points(points):
	# Creation of normalization homography for 8-point method
	mean = np.mean(points, axis=0)
	std_dev = np.std(points, axis=0)
	scale = np.sqrt(2) / std_dev

	T = np.array([
		[scale[0], 0, -scale[0] * mean[0]],
		[0, scale[1], -scale[1] * mean[1]],
		[0, 0, 1]
	])

	normalized_points = np.dot(T, np.column_stack((points, np.ones(len(points)))).T).T

	return normalized_points[:, :2], T

def estimate_F(image1_points, image2_points):
	# Normalize points
	points1_normalized, T1 = normalize_points(image1_points)
	points2_normalized, T2 = normalize_points(image2_points)

	# Construct A matrix
	A = np.array([
		[x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
		for (x1, y1), (x2, y2) in zip(points1_normalized, points2_normalized)
	])

	# SVD to solve for fundamental matrix parameters
	_, _, Vt = np.linalg.svd(A) # Using SVD to solve for homography terms
	F_normalized = Vt[-1].reshape((3, 3))

	# Condition F to be rank 2
	F_conditioned = condition_F(F_normalized)

	F = T2.T @ F_conditioned @ T1

	F /= F[2, 2]

	return F


def estimate_epipoles(F):
	# Epipoles are left and right null vectors of F
	e = null_space(F)
	e_p = null_space(F.T)

	e /= e[-1]
	e_p /= e_p[-1]

	return e, e_p


def calculate_camera_matrices(F, e_p):
	e_p = e_p.flatten()

	# Skew symmetric matrix for construction of P'
	s = np.array([[0, -e_p[2], e_p[1]], [e_p[2], 0, -e_p[0]], [-e_p[1], e_p[0], 0]])

	P_p = np.hstack((s @ F, e_p.reshape(-1, 1)))

	return P_p

def compute_rectification_homography_right(right_epipole):
	# Translational homographies to bring epipole to image origin and back
	T1 = np.array([[1, 0, -image2_width / 2], [0, 1, -image2_height / 2], [0, 0, 1]])
	T2 = np.array([[1, 0, image2_width / 2], [0, 1, image2_height / 2], [0, 0, 1]])

	right_epipole = (T1 @ right_epipole).ravel()

	right_epipole = right_epipole / right_epipole[2]

	# Calculate angle to bring epipole to infinity
	theta = -np.arctan2(right_epipole[1], right_epipole[0])

	R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

	transformed_epipole = (R @ right_epipole.reshape((3, 1))).ravel()

	transformed_epipole = transformed_epipole / transformed_epipole[2]

	f = transformed_epipole[0]

	G = np.array([[1, 0, 0], [0, 1, 0], [-1 / f, 0, 1]])

	H_p = T2 @ G @ R @ T1

	H_p /= H_p[2, 2]

	return H_p

def compute_rectification_homography_left(H_p, left_epipole, image1_points, image2_points):
	# Initial estimate for left homography
	H_hat = compute_rectification_homography_right(left_epipole)

	A = []
	b = []

	# Use minimization technique to find left homography
	for point_left, point_right in zip(image1_points, image2_points):
		point_left_hc = HC(point_left)
		point_right_hc = HC(point_right)

		transformed_point_left = (H_hat @ point_left_hc).flatten()
		transformed_point_right = (H_p @ point_right_hc).flatten()

		A.append([transformed_point_left[0], transformed_point_left[1], 1])
		b.append(transformed_point_right[0])


	A = np.array(A)
	b = np.array(b)

	min_solution = np.linalg.pinv(A) @ b

	Ha = np.array([[min_solution[0], min_solution[1], min_solution[2]], [0, 1, 0], [0, 0, 1]])

	H = Ha @ H_hat

	H /= H[2, 2]

	return H

# def verify_F(F):
# 	for point1, point2 in zip(image1_points, image2_points):
# 		point1_hc = HC(point1)
# 		point2_hc = HC(point2)

# 		print("x'^T * F * x: ", point2_hc.T @ F @ point1_hc)

def triangulate_point(P_p, point1, point2):
	# Standard triangulation using P' and P
	A = [
		point1[0] * P[2] - P[0],
		point1[1] * P[2] - P[1],
		point2[0] * P_p[2] - P_p[0],
		point2[1] * P_p[2] - P_p[1]
	]
	A = np.array(A)
	_, _, Vt = np.linalg.svd(A.T @ A)
	X = Vt[-1]

	return X / X[3]

def backprojection(P_p, image1_points, image2_points):
	errors = []
	world_points = []

	for point1, point2 in zip(image1_points, image2_points):
		X = triangulate_point(P_p, HC(point1), HC(point2))  # Triangulate the 3D point
		world_points.append(X.flatten()[:3])

		# Apply corresponding camera matrix to world point
		reprojected_point1 = P @ X
		reprojected_point2 = P_p @ X

		# Normalize the projected points
		reprojected_point1 /= reprojected_point1[2]
		reprojected_point2 /= reprojected_point2[2]

		# Calculate the Euclidean distance (geometric error)
		error1 = np.linalg.norm(reprojected_point1[:2] - point1)**2
		error2 = np.linalg.norm(reprojected_point2[:2] - point2)**2
		
		errors.extend([error1, error2])
	
	return np.array(errors), np.array(world_points)

def geometric_error(P_p_vector, image1_points, image2_points):

	P_p = P_p_vector.reshape(3, 4)  # Reshape the flattened P' vector

	errors, _ = backprojection(P_p, image1_points, image2_points)

	return errors

def refine_P_p(P_p, image1_points, image2_points):

	P_p_vector = P_p.flatten()  # Flatten P' for optimization

	# Perform LM on P_p to reduce euclidean distance error
	result = least_squares(
		geometric_error,
		P_p_vector,
		method='lm',
		args=(image1_points, image2_points)
	)

	refined_P_p = result.x.reshape(3, 4)  # Reshape optimized vector back to matrix

	return refined_P_p

def rectify_images(image1, image2, image1_points, image2_points, collect_world_points):

	# Estimate F from point correspondences
	F = estimate_F(image1_points, image2_points)

	# Calculate initial estimates for epipoles
	initial_e, initial_e_p = estimate_epipoles(F)
	initial_e = initial_e.reshape((3, 1))
	initial_e_p = initial_e_p.reshape((3, 1))

	# Use F and epipole estimates to construct P'
	initial_P_p = calculate_camera_matrices(F, initial_e_p)

	# Use LM to refine P' estimate
	refined_P_p = refine_P_p(initial_P_p, image1_points, image2_points)

	# Extract new epipole from P', construct skew-symmetric matrix for refined F calculation
	refined_e_p = refined_P_p[:, 3]
	s = np.array([[0, -refined_e_p[2], refined_e_p[1]], [refined_e_p[2], 0, -refined_e_p[0]], [-refined_e_p[1], refined_e_p[0], 0]])

	# Calculate refined F and condition to enforce rank 2
	refined_F = s @ refined_P_p @ np.linalg.pinv(P)

	refined_F /= refined_F[2, 2]
	refined_F = condition_F(refined_F)

	# Extract refined epipoles from refined F
	refined_e, refined_e_p = estimate_epipoles(refined_F)

	# Construct rectification homographies from 
	H_p = compute_rectification_homography_right(refined_e_p)
	H = compute_rectification_homography_left(H_p, refined_e, image1_points, image2_points)

	if collect_world_points:
		_, world_points = backprojection(refined_P_p, image1_points, image2_points)
		return H, H_p, refined_P_p, world_points

	return H, H_p, refined_P_p

def create_visualization(image1_rectified, image2_rectified, H, H_p):
	combined_rectified_image = np.hstack((image1_rectified, image2_rectified))

	for (x1, y1), (x2, y2) in zip(image1_points, image2_points):
		point_left_hc = HC((x1, y1))
		point_right_hc = HC((x2, y2))

		transformed_point_left = (H @ point_left_hc).flatten()
		transformed_point_right = (H_p @ point_right_hc).flatten()

		pixel_left = (int(transformed_point_left[0] / transformed_point_left[2]), int(transformed_point_left[1] / transformed_point_left[2]))
		pixel_right = (int(transformed_point_right[0] / transformed_point_right[2]), int(transformed_point_right[1] / transformed_point_right[2]))


		cv2.circle(image1_rectified, pixel_left, radius=5, color=(255, 0, 0), thickness=-1)
		cv2.circle(image2_rectified, pixel_right, radius=5, color=(0, 255, 0), thickness=-1)

		cv2.circle(combined_rectified_image, pixel_left, radius=5, color=(255, 0, 0), thickness=-1)
		cv2.circle(combined_rectified_image, (pixel_right[0] + new_width, pixel_right[1]), radius=5, color=(0, 255, 0), thickness=-1)

		cv2.line(combined_rectified_image, pixel_left, (pixel_right[0] + new_width, pixel_right[1]), color=(0, 0, 255), thickness=2)

	return combined_rectified_image

def extract_interest_points(rectified_image, canny_threshold1, canny_threshold2):
	# Extract edges using canny detector, collect points where mask is true
	edges = cv2.Canny(rectified_image, canny_threshold1, canny_threshold2)
	points = np.column_stack(np.where(edges > 0))

	# Sort by row
	points = points[np.lexsort((points[:, 1], points[:, 0]))]

	if visualize:
		cv2.imshow("Interest Points", edges)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return points

def get_candidate_points(point, interest_points2, row_window=3):
	row_range = range(point[0] - row_window, point[0] + row_window + 1)
	return interest_points2[np.isin(interest_points2[:, 0], row_range)]

def extract_patch(image, point, patch_size=5):
	half_size = patch_size // 2
	x, y = point
	h, w = image.shape[:2]

	# Out of bounds checking
	if(x - half_size < 0 or x + half_size >= w or y - half_size < 0 or y > half_size >= h):
		return None

	# Collect all points within patch window
	return image[x - half_size:x + half_size + 1, y - half_size:y + half_size + 1]

def SSD(patch1, patch2):
	# Sum of squared differences if patches are both all full
	if patch1 is None or patch2 is None:
		return float("inf")
	return np.sum((patch1 - patch2)**2)

def compute_dense_correspondences(image1_rectified, image2_rectified, interest_points1, interest_points2):
	point1_correspondences = []
	point2_correspondences = []

	for point1 in interest_points1:
		# Find potential candidates through buffered row
		candidates = get_candidate_points(point1, interest_points2)
		if len(candidates) == 0:
			continue

		# Go through each candidate and find lowest SSD between parent patch and potential patch
		patch1 = extract_patch(image1_rectified, point1)
		best_candidate, best_score = None, float("inf")

		for candidate in candidates:
			patch2 = extract_patch(image2_rectified, candidate)
			score = SSD(patch1, patch2)

			if score < best_score:
				best_score = score
				best_candidate = candidate

		# Add best candidate to correspondence list
		if best_candidate is not None:
			point1_correspondences.append([point1[1], point1[0]])
			point2_correspondences.append([best_candidate[1], best_candidate[0]])

	return np.array(point1_correspondences), np.array(point2_correspondences)

if __name__ == "__main__":
	image1_points = np.array([[602, 68], [287, 64], [496, 304], [303, 506], [611, 345], [195, 446], [284, 475], [548, 233]])
	image2_points = np.array([[584, 83], [268, 80], [505, 311], [277, 490], [598, 368], [185, 425], [263, 459], [533, 250]])

	image1_with_points = image1.copy()
	image2_with_points = image2.copy()

	if visualize:
		for (x1, y1), (x2, y2) in zip(image1_points, image2_points):
			cv2.circle(image1_with_points, (x1, y1), radius=5, color=(0, 0, 255), thickness=-1)
			cv2.circle(image2_with_points, (x2, y2), radius=5, color=(0, 0, 255), thickness=-1)

		cv2.imshow("image1 with points", image1_with_points)
		cv2.imshow("image2 with points", image2_with_points)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	# Calculate rectification homographies
	H, H_p, _ = rectify_images(image1, image2, image1_points, image2_points, False)

	# Calculate updated buffers from homographies
	width_new1, height_new1 = calculate_raster(image1, H)
	width_new2, height_new2 = calculate_raster(image2, H_p)


	if visualize:
		new_width = max(width_new1, width_new2)
		new_height = max(height_new1, height_new2)

		image1_rectified = cv2.warpPerspective(image1, H, (new_width, new_height))
		image2_rectified = cv2.warpPerspective(image2, H_p, (new_width, new_height))
	else:
		image1_rectified = cv2.warpPerspective(image1, H, (width_new1, height_new1))
		image2_rectified = cv2.warpPerspective(image2, H_p, (width_new2, height_new2))

	# if visualize:
	# 	combined_image = create_visualization(image1_rectified, image2_rectified, H, H_p)

	# 	cv2.imshow("rectified left image", image1_rectified)
	# 	cv2.imshow("rectified right image", image2_rectified)
	# 	cv2.imshow("combined_image", combined_image)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()


	# Canny edge detection parameters
	canny_threshold1 = 600
	canny_threshold2 = 800

	# Extract interest points
	interest_points1 = extract_interest_points(image1_rectified, canny_threshold1, canny_threshold2)
	interest_points2 = extract_interest_points(image2_rectified, canny_threshold1, canny_threshold2)

	# Find correspondences using SSD patch scoring
	new_image1_points, new_image2_points = compute_dense_correspondences(image1_rectified, image2_rectified, interest_points1, interest_points2)

	# Refine P' from dense correspondences
	_, _, refined_P_p, world_points = rectify_images(image1_rectified, image2_rectified, new_image1_points, new_image2_points, True)

	if visualize:
		combined_rectified_image = np.hstack((image1_rectified, image2_rectified))

		for point1, point2 in zip(new_image1_points, new_image2_points):
			color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

			offset = image1_rectified.shape[1]
			point2 = (point2[0] + offset, point2[1])

			cv2.circle(combined_rectified_image, point1, radius=2, color=(0, 255, 0), thickness=-1)
			cv2.circle(combined_rectified_image, point2, radius=2, color=(0, 255, 0), thickness=-1)

			cv2.line(combined_rectified_image, point1, point2, thickness=2, color=color)

		cv2.imshow("combined_image", combined_rectified_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# Populate 2D and 3D points to form pointcloud
	visualization_points_3d = []
	visualization_points_2d_left = []
	visualization_points_2d_right = []

	for (x1, y1), (x2, y2) in zip(image1_points, image2_points):
		point_left_hc = HC((x1, y1))
		point_right_hc = HC((x2, y2))

		transformed_point_left = (H @ point_left_hc).flatten()
		transformed_point_right = (H_p @ point_right_hc).flatten()

		visualization_points_2d_left.append([transformed_point_left[0] / transformed_point_left[2], transformed_point_left[1] / transformed_point_left[2]])
		visualization_points_2d_right.append([transformed_point_right[0] / transformed_point_right[2], transformed_point_right[1] / transformed_point_right[2]])

		world_visualization_point = triangulate_point(refined_P_p, transformed_point_left / transformed_point_left[2], transformed_point_right / transformed_point_right[2])
		visualization_points_3d.append(world_visualization_point)


	visualization_points_3d = np.array(visualization_points_3d)
	
	print("world_points shape: ", world_points.shape)
	print("visualization_points shape: ", visualization_points_3d.shape)

	# Plotting
	X = world_points[:, 0]
	Y = world_points[:, 1]
	Z = world_points[:, 2]

	X_v = visualization_points_3d[:, 0]
	Y_v = visualization_points_3d[:, 1]
	Z_v = visualization_points_3d[:, 2]

	fig = plt.figure(figsize=(6, 9))

	ax1 = fig.add_subplot(312, projection='3d')
	ax2 = fig.add_subplot(311)
	ax3 = fig.add_subplot(313)

	ax1.scatter(X, Y, Z, c='b', s=10)
	ax1.scatter(X_v, Y_v, Z_v, c='r', s=100)
	ax1.set_xlabel("X")
	ax1.set_ylabel("Y")
	ax1.set_zlabel("Z")
	ax1.set_title("3D Projection")


	ax2.imshow(image1_rectified)
	ax3.imshow(image2_rectified)

	for i in range(len(visualization_points_3d)):
		left_point = visualization_points_2d_left[i]
		right_point = visualization_points_2d_right[i]

		ax2.scatter(left_point[0], left_point[1], color='r', s=50)
		ax3.scatter(right_point[0], right_point[1], color='r', s=50)


	ax2.set_xlabel('x')
	ax2.set_ylabel('y')
	ax2.set_title("Left Rectified Image")

	ax3.set_xlabel('x')
	ax3.set_ylabel('y')
	ax3.set_title("Right Rectified Image")


	plt.tight_layout()
	plt.show()




