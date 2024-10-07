import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import math

img1_points = np.array([[923, 1300], [2330, 1165], [2241, 2134], [1006, 2668]])
img2_points = np.array([[655, 1601], [1605, 1255], [1632, 2438], [627, 2458]])
img3_points = np.array([[1285, 1057], [2504, 2244], [1754, 2842], [640, 1818]])
test_image_points = np.array([[0, 0], [783, 0], [783, 665], [0, 665]])

bill1_points = np.array([[849, 682], [2564, 919], [2276, 2884], [796, 3155]])
bill2_points = np.array([[708, 1522], [2241, 1497], [2046, 2710], [957, 2714]])
bill3_points = np.array([[730, 1014], [2266, 1116], [2037, 2882], [821, 2638]])
curiosity_points = np.array([[0, 0], [2000, 0], [2000, 2757], [0, 2757]])

img1 = np.array(cv2.imread('HW2_images/img1.jpg'))
img2 = np.array(cv2.imread('HW2_images/img2.jpg'))
img3 = np.array(cv2.imread('HW2_images/img3.jpg'))
test_image = np.array(cv2.imread('HW2_images/alex_honnold.jpg'))


bill_image1 = np.array(cv2.imread('bill_johnson1.jpg'))
bill_image2 = np.array(cv2.imread('bill_johnson2.jpg'))
bill_image3 = np.array(cv2.imread('bill_johnson3.jpg'))
curiosity_image = np.array(cv2.imread('curiosity.jpg'))



def find_homgoraphy(domain_set, range_set):
	num_points = domain_set.shape[0]

	A = []
	p = []

	for i in range(num_points):
		# Convert to HC
		u, v, w = [domain_set[i][0], domain_set[i][1], 1]
		up, vp, wp = [range_set[i][0], range_set[i][1], 1]

		# Set up A and p matrices
		A.append([u, v, w, 0, 0, 0, -up * u, -up * v])
		A.append([0, 0, 0, u, v, w, -vp * u, -vp * v])
		p.append(up)
		p.append(vp)

	A = np.array(A)
	p = np.array(p)

	H = np.dot(np.linalg.inv(A), p.T)
	H = np.append(H, 1)
	H = np.reshape(H, (3, 3))

	return(H)

def mask_frame(image, points):
	mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

	cv2.fillPoly(mask, [points], 255)

	return mask

def find_affine_homgoraphy(domain_set, range_set):
	num_points = domain_set.shape[0]

	A = []
	p = []

	for i in range(num_points):
		# Convert to HC
		u, v, w = [domain_set[i][0], domain_set[i][1], 1]
		up, vp, wp = [range_set[i][0], range_set[i][1], 1]

		# Set up A and p matrices
		A.append([u, v, w, 0, 0, 0])
		A.append([0, 0, 0, u, v, w])
		p.append(up)
		p.append(vp)

	A = np.array(A)
	p = np.array(p)

	H = np.dot(np.linalg.pinv(A), p.T) # taking pseudoinverse due to non-square A matrix
	H = np.append(H, 0)
	H = np.append(H, 0)
	H = np.append(H, 1)
	H = np.reshape(H, (3, 3))

	return(H)

def apply_homography_with_inverse(src_image, H, dest_image, frame_points):

	H_inv = np.linalg.inv(H)

	H_inv /= H_inv[2, 2]

	domain_height, domain_width, _ = src_image.shape
	range_height, range_width, _ = dest_image.shape

	domain_image = np.zeros((domain_height, domain_width, 3), dtype=np.uint8)

	frame_mask = mask_frame(dest_image, frame_points)

	for v in range(frame_mask.shape[0]):
		for u in range(frame_mask.shape[1]):
			if(frame_mask[v, u] == 255):
				range_point = np.array([u, v, 1])
				domain_point = H_inv @ range_point

				domain_point /= domain_point[2]

				domain_u, domain_v = int(domain_point[0]), int(domain_point[1])

				if(0 <= domain_u < domain_width and 0 <= domain_v < domain_height):
					dest_image[v, u] = src_image[domain_v, domain_u]

				
	return dest_image

def apply_homography_direct(src_image, H, dest_image, put_on_image):

	domain_height, domain_width, _ = src_image.shape
	range_height, range_width, _ = dest_image.shape

	if(not put_on_image):
		dest_image = np.zeros((range_height, range_width, 3), dtype=np.uint8)

	for v in range(domain_height):
		for u in range(domain_width):
			domain_point = np.array([u, v, 1])
			range_point = H @ domain_point

			range_point /= range_point[2]

			range_u, range_v = int(range_point[0]), int(range_point[1])

			if(0 <= range_u < range_width and 0 <= range_v < range_height):
				dest_image[range_v, range_u] = src_image[v, u]

				
	return dest_image


def combo_homography(img1, img2, img3):
	H12 = find_homgoraphy(img1_points, img2_points)
	H23 = find_homgoraphy(img2_points, img3_points)

	H13 = H12 @ H23

	img1to3 = apply_homography_direct(img1, H13, img3, False)

	return img1to3

# plt.imshow(bill_image2)
# plt.show()

# plt.imshow(bill_image3)
# plt.show()

# time.sleep(100)

## Finding Homographies and Affine Homographies
# H1 = find_homgoraphy(test_image_points, img1_points)
# H2 = find_homgoraphy(test_image_points, img2_points)
# H3 = find_homgoraphy(test_image_points, img3_points)
H1 = find_homgoraphy(curiosity_points, bill1_points)
H2 = find_homgoraphy(curiosity_points, bill2_points)
H3 = find_homgoraphy(curiosity_points, bill3_points)

# A1 = find_affine_homgoraphy(test_image_points, img1_points)
# A2 = find_affine_homgoraphy(test_image_points, img2_points)
# A3 = find_affine_homgoraphy(test_image_points, img3_points)

## Uncomment for direct homography approach
# transformed_image1 = apply_homography_direct(test_image, H1, img1, True)
# transformed_image2 = apply_homography_direct(test_image, H2, img2, True)
# transformed_image3 = apply_homography_direct(test_image, H3, img3, True)

## Uncomment for affine transformations
# transformed_image1 = apply_homography_with_inverse(test_image, A1, img1, img1_points)
# transformed_image2 = apply_homography_with_inverse(test_image, A2, img2, img2_points)
# transformed_image3 = apply_homography_with_inverse(test_image, A3, img3, img3_points)

## Uncomment for best result transformations
# transformed_image1 = apply_homography_with_inverse(test_image, H1, img1, img1_points)
# transformed_image2 = apply_homography_with_inverse(test_image, H2, img2, img2_points)
# transformed_image3 = apply_homography_with_inverse(test_image, H3, img3, img3_points)
transformed_image1 = apply_homography_with_inverse(curiosity_image, H1, bill_image1, bill1_points)
transformed_image2 = apply_homography_with_inverse(curiosity_image, H2, bill_image2, bill2_points)
transformed_image3 = apply_homography_with_inverse(curiosity_image, H3, bill_image3, bill3_points)


## Uncomment for combination homography
# transformed_image1 = combo_homography(img1, img2, img3)

transformed_image1 = cv2.cvtColor(transformed_image1, cv2.COLOR_BGR2RGB)
transformed_image2 = cv2.cvtColor(transformed_image2, cv2.COLOR_BGR2RGB)
transformed_image3 = cv2.cvtColor(transformed_image3, cv2.COLOR_BGR2RGB)

plt.imshow(transformed_image1)
plt.show()

plt.imshow(transformed_image2)
plt.show()

plt.imshow(transformed_image3)
plt.show()


