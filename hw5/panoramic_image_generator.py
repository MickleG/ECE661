import cv2
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

testname = "fountain"

# Loading images
img1 = cv2.imread('pics/' + testname + '/1.jpg')
img2 = cv2.imread('pics/' + testname + '/2.jpg')
img3 = cv2.imread('pics/' + testname + '/3.jpg')
img4 = cv2.imread('pics/' + testname + '/4.jpg')
img5 = cv2.imread('pics/' + testname + '/5.jpg')

save_images = True

# Helper function to convert point to homogeneous coordinates
def HC(point):
	return np.array([point[0], point[1], 1])

def estimate_homography(pairs):
	A = []
	for (x1, y1), (x2, y2) in pairs:
		A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
		A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])

	A = np.array(A)

	_, _, V = np.linalg.svd(A) # Using SVD to solve for homography terms
	H = V[-1].reshape((3, 3))

	return H / H[2, 2]

def calculate_raster(Hleft, Hright, img_left, img_right):
	img_left_height, img_left_width, _ = img_left.shape
	img_right_height, img_right_width, _ = img_right.shape
	
	# Transforming the leftmost and rightmost images to determined final raster bounds
	corner_top_left = HC([0, 0])
	corner_top_right = HC([img_right_width, 0])
	corner_bottom_right = HC([img_right_width, img_right_height])
	corner_bottom_left = HC([0, img_left_height])

	corners_left_hc = np.array([corner_top_left, corner_bottom_left]).T
	corners_right_hc = np.array([corner_top_right, corner_bottom_right]).T

	transformed_left = np.matmul(Hleft, corners_left_hc)
	transformed_right = np.matmul(Hright, corners_right_hc)

	transformed_left = transformed_left / (transformed_left[-1, :] + 1e-6)
	transformed_right = transformed_right / (transformed_right[-1, :] + 1e-6)

	# Choosing min_x based on transformed leftmost image corners and max_x based on rightmost transformed image corners. Same idea with y
	min_x = int(min(transformed_left[0,:]))
	max_x = int(max(transformed_right[0,:]))
	min_y = int(min(min(transformed_left[1,:]), min(transformed_right[1,:])))
	max_y = int(max(max(transformed_left[1,:]), max(transformed_right[1,:])))

	w_out = max_x - min_x
	h_out = max_y - min_y

	H_trans = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

	return w_out, h_out, H_trans

# Helper function to apply homography to a single point
def apply_homography_point(H, point):
	point_hc = HC(point)

	transformed_point = np.dot(H, point_hc)

	return (transformed_point[0] / transformed_point[2], transformed_point[1] / transformed_point[2])
	
def apply_homography_with_inverse(src_image, H, dest_image, Htrans):

	H = np.matmul(Htrans, H)

	H_inv = np.linalg.inv(H)

	H_inv /= H_inv[2, 2]

	domain_height, domain_width, _ = src_image.shape
	range_height, range_width, _ = dest_image.shape

	domain_image = np.zeros((domain_height, domain_width, 3), dtype=np.uint8)

	# Sampling range space to directly place domain image rgb values using the inverse homography - this method prevents transparency issues that are found in the direct homography approach
	for v in range(range_height):
		for u in range(range_width):
			range_point = np.array([u, v, 1])
			domain_point = H_inv @ range_point

			domain_point /= domain_point[2]

			domain_u, domain_v = int(domain_point[0]), int(domain_point[1])

			if(0 <= domain_u < domain_width and 0 <= domain_v < domain_height):
				dest_image[v, u] = src_image[domain_v, domain_u]

	return dest_image

def sift(img1, img2, namestring):

	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	combined_image = np.concatenate((img1, img2), axis = 1)

	# Instantiating sift object, creating keypoints and descriptors
	sift = cv2.SIFT_create()
	keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
	keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

	# Using k-nearest neighbors for matching, and applying Lowe Ratio test to determine if match is strong
	matcher = cv2.BFMatcher()

	matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

	good_matches = []

	# Lowe ratio test to determine good matches
	for m, n in matches:
		if m.distance < 0.75 * n.distance:
			pt1 = keypoints1[m.queryIdx].pt
			pt2 = keypoints2[m.trainIdx].pt
			good_matches.append((pt1, pt2))

	# Draw lines between matches
	for match in good_matches:
		(x1, y1) = match[0]
		(x2, y2) = match[1]
		
		x2 += img1.shape[1]
		
		cv2.circle(combined_image, (int(x1), int(y1)), 2, color=(0, 0, 255), thickness=-1)
		cv2.circle(combined_image, (int(x2), int(y2)), 2, color=(0, 0, 255), thickness=-1)
		
		cv2.line(combined_image, (int(x1), int(y1)), (int(x2), int(y2)), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=1)

	if(save_images):
		cv2.imwrite('pics/' + testname + '/sift_' + namestring + '.jpg', combined_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

	return good_matches


def ransac(matches, epsilon, p, delta, img1, img2, namestring):
	
	best_inliers = []
	best_H = None
	n = 8 # Minimum number of correspondences to find a homography
	n_total = len(matches)
	

	combined_image = np.concatenate((img1, img2), axis = 1) # generation of combined image for visualization

	if len(matches) < 4: # Rejecting sift results with less than the minimum correspondencies for a homography
		print("not enough correspondences to run RANSAC, please adjust SIFT parameters")
		return None, None

	N = math.ceil(math.log(1 - p) / math.log(1 - (1 - epsilon) ** n)) # Finding number of iterations by solving for N in the relation 1-[1-(1-epsilon)^n]^N = p
	M = (1 - epsilon) * n_total # calculating acceptable number of outliers for faster computation speed

	# print("Number of iterations: ", N)

	for i in range(N):
		# Sample random set of n matches from sift and generate homography
		sampled_matches = random.sample(matches, n)

		pairs = [((n[0][0], n[0][1]), (n[1][0], n[1][1])) for n in sampled_matches]

		H = estimate_homography(pairs)

		inliers = []

		# Applying homography from random sample to all matches to determine inlier consensus
		for match in matches:
			(x1, y1) = match[0]
			(x2, y2) = match[1]

			projected_point = apply_homography_point(H, (x1, y1))

			distance = np.linalg.norm(np.array(projected_point) - np.array([x2, y2]))

			if distance < delta:
				inliers.append(match)

		# Test for if current estimation has better inlier consensus
		if len(inliers) > len(best_inliers):
			best_inliers = inliers
			best_H = H

		if(len(best_inliers)) > M: # if M is reached before max_iterations N, then break as this is acceptable
			break


	# Draw lines between matches
	for match in matches:
		(x1, y1) = match[0]
		(x2, y2) = match[1]
		
		x2 += img1.shape[1]
		
		if match in best_inliers:
			cv2.circle(combined_image, (int(x1), int(y1)), 2, color=(0, 255, 0), thickness=-1)
			cv2.circle(combined_image, (int(x2), int(y2)), 2, color=(0, 255, 0), thickness=-1)
			
			cv2.line(combined_image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=1)
		else:
			cv2.circle(combined_image, (int(x1), int(y1)), 2, color=(0, 0, 255), thickness=-1)
			cv2.circle(combined_image, (int(x2), int(y2)), 2, color=(0, 0, 255), thickness=-1)
			
			cv2.line(combined_image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=1)


	if(save_images):
		cv2.imwrite('pics/' + testname + '/ransac_' + namestring + '.jpg', combined_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

	return best_H, best_inliers

def nonlinear_least_squares(H_initial, inliers):

	# functions used for LM in scipy
	def homography_transform(H, points):
		H = H.reshape(3, 3)
		points_h = np.hstack([points, np.ones((points.shape[0], 1))]) # Convert to homogeneous coordinates
		points_transformed_h = np.dot(H, points_h.T).T # Apply homography
		points_transformed_h /= points_transformed_h[:, 2].reshape(-1, 1) # Normalize by the last (homogeneous) coordinate
		return points_transformed_h[:, :2]

	def reprojection_error(H, points1, points2):
		points1_proj = homography_transform(H, points1) # Transform points from image 1 using H
		errors = points1_proj - points2  # Calculate the reprojection error
		return errors.ravel()


	points1 = np.array([inlier[0] for inlier in inliers]) # Points from image 1
	points2 = np.array([inlier[1] for inlier in inliers]) # Corresponding points from image 2


	H_initial_flat = H_initial.flatten()

	# Use scipy.optimize.least_squares with the LM argument
	result = least_squares(reprojection_error, H_initial_flat, args=(points1, points2), method='lm')

	# Reshape the optimized homography back into a 3x3 matrix
	H_refined = result.x.reshape(3, 3)

	return H_refined / H_refined[2, 2] # Divide by last term


def create_panorama(img1, img2, img3, img4, img5, H12, H23, H34, H45):
	
	# Calculating homographies that convert images to middle image (in this case img3)
	H13 = H23 @ H12
	H43 = np.linalg.inv(H34)
	H54 = np.linalg.inv(H45)
	H53 = H43 @ H54

	# Calculate size of output raster for panorama
	panorama_width, panorama_height, Htrans = calculate_raster(H13, H53, img1, img5)

	panorama_img = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

	# Directly apply img3 to panorama img with just translation homography to serve as starting point for other image projections
	for v in range(img3.shape[0]):
		for u in range(img3.shape[1]):
			point_hc = np.array([u, v, 1])

			transformed_point_hc = Htrans @ point_hc
			transformed_point_hc = transformed_point_hc / transformed_point_hc[2]

			panorama_u, panorama_v = int(transformed_point_hc[0]), int(transformed_point_hc[1])

			if(0 <= panorama_v < panorama_height and 0 <= panorama_u < panorama_width):
				panorama_img[panorama_v, panorama_u] = img3[v, u]

	img3 = panorama_img

	# Apply homography from each image to image 3 using their corresponding homographies, mutating the panorama along the way
	img3 = apply_homography_with_inverse(img1, H13, img3, Htrans)
	img3 = apply_homography_with_inverse(img2, H23, img3, Htrans)
	img3 = apply_homography_with_inverse(img4, H43, img3, Htrans)
	img3 = apply_homography_with_inverse(img5, H53, img3, Htrans)

	if(save_images):
		cv2.imwrite('pics/' + testname + '/panorama_before_LM.jpg', img3, [cv2.IMWRITE_JPEG_QUALITY, 90])
	else:
		img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
		plt.imshow(img3)
		plt.show()

# Step 1, extracting interest points pairwise between adjacent images
matches12 = sift(img1=img1, img2=img2, namestring="12")
matches23 = sift(img1=img2, img2=img3, namestring="23")
matches34 = sift(img1=img3, img2=img4, namestring="34")
matches45 = sift(img1=img4, img2=img5, namestring="45")

# Step 2, performing outlier rejection with RANSAC
H12, inliers12 = ransac(matches=matches12, epsilon=0.35, p=0.99, delta=3, img1=img1, img2=img2, namestring="12")
H23, inliers23 = ransac(matches=matches23, epsilon=0.35, p=0.99, delta=3, img1=img2, img2=img3, namestring="23")
H34, inliers34 = ransac(matches=matches34, epsilon=0.35, p=0.99, delta=3, img1=img3, img2=img4, namestring="34")
H45, inliers45 = ransac(matches=matches45, epsilon=0.35, p=0.99, delta=3, img1=img4, img2=img5, namestring="45")

# Step 3, Nonlinear Least-Squares Homography Refinement
H12 = nonlinear_least_squares(H12, inliers12)
H23 = nonlinear_least_squares(H23, inliers23)
H34 = nonlinear_least_squares(H34, inliers34)
H45 = nonlinear_least_squares(H45, inliers45)

# Step 4, applying homographies to common reference frame
create_panorama(img1=img1, img2=img2, img3=img3, img4=img4, img5=img5, H12=H12, H23=H23, H34=H34, H45=H45)







