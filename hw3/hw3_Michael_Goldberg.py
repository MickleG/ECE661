import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import math


img1 = np.array(cv2.imread('HW3_images/board_1.jpeg'))
img2 = np.array(cv2.imread('HW3_images/corridor.jpeg'))

test_img1 = np.array(cv2.imread('HW3_images/test_img1.jpg'))
test_img2 = np.array(cv2.imread('HW3_images/test_img2.jpg'))

## Uncomment for pixel location visualization
# plt.imshow(img1)
# plt.show()

# plt.imshow(test_img2)
# plt.show()


width1 = 800
height1 = 1200
width2 = 300
height2 = 600


img1_points = np.array([[71, 421], [1220, 140], [1354, 1950], [421, 1789]])
test_img1_points = np.array([[923, 1300], [2330, 1165], [2241, 2134], [1006, 2668]])
test_img2_points = np.array([[730, 1014], [2266, 1116], [2037, 2882], [821, 2638]])


img1_2step_points = np.array([[71, 421], [1220, 140], [1354, 1950], [71, 421], [421, 1789], [1354, 1950]])
test_img1_2step_points = np.array([[923, 1300], [2330, 1165], [2241, 2134], [923, 1300], [1006, 2668], [2241, 2134]])
test_img2_2step_points = np.array([[730, 1014], [2266, 1116], [2037, 2882], [730, 1014], [821, 2638], [2037, 2882]])


img1_1step_points = np.array([[71, 421], [1220, 140], [1354, 1950], [71, 421], [421, 1789], [1354, 1950], [880, 539], [981, 523], [998, 637], [517, 1076], [528, 1165], [612, 1158], [503, 603], [572, 484], [682, 572]])
img2_1step_points = np.array([[1083, 528], [1306, 487], [1296, 1340], [916, 1127], [811, 1069], [814, 576], [1312, 1357], [1327, 496], [1695, 427], [1087, 345], [1081, 521], [1308, 472], [1699, 107], [1693, 392], [1323, 470]])



transformed_img1_points = np.array([[0, 0], [width1, 0], [width1, height1], [0, height1]])
transformed_img2_points = np.array([[0, 0], [width2, 0], [width2, height2], [0, height2]])

transformed_test_img1_points = np.array([[0, 0], [height1, 0], [height1, width1], [0, width1]])
transformed_test_img2_points = np.array([[0, 0], [width1, 0], [width1, height1], [0, height1]])


def HC(point):
	return [point[0], point[1], 1.001]

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


def apply_homography_with_inverse(src_img, H):

	dest_img, H = calculate_raster_and_resize(src_img, H)


	domain_height, domain_width, _ = src_img.shape
	range_height, range_width, _ = dest_img.shape


	H_inv = np.linalg.inv(H)

	H_inv /= H_inv[2, 2]

	for v in range(dest_img.shape[0]):
		for u in range(dest_img.shape[1]):
			range_point = np.array([u, v, 1])
			domain_point = H_inv @ range_point

			domain_point /= domain_point[2]

			domain_u, domain_v = int(domain_point[0]), int(domain_point[1])

			if(0 <= domain_u < domain_width and 0 <= domain_v < domain_height):
				dest_img[v, u] = src_img[domain_v, domain_u]

				
	return dest_img

def calculate_raster(img, H):
	img_height, img_width, _ = img.shape

	corners = np.array([[0, 0, 1], [img_width, 0, 1], [img_width, img_height, 1], [0, img_height, 1]])
	corners_hc = corners.T

	corners_transformed_hc = np.matmul(H, corners_hc)
	corners_transformed_hc = corners_transformed_hc / (corners_transformed_hc[-1, :] + 1e-6)


	min_x = int(min(corners_transformed_hc[0,:]))
	max_x = int(max(corners_transformed_hc[0,:]))
	min_y = int(min(corners_transformed_hc[1,:]))
	max_y = int(max(corners_transformed_hc[1,:]))

	# print("min_x: ", min_x)
	# print("max_x: ", max_x)

	# print("min_y: ", min_y)
	# print("max_y: ", max_y)

	w_out = max_x - min_x
	h_out = max_y - min_y

	H_trans = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

	H = np.matmul(H_trans, H)

	return w_out, h_out, H, min_x, max_x

def calculate_raster_and_resize(img, H):

	# print(H)

	w_out, h_out, _, _, _ = calculate_raster(img, H)

	#resizing
	aspect_ratio = w_out / (h_out + 1e-6)
	h_out_final = 700
	w_out_final = aspect_ratio * h_out_final
	H_resize = np.array([[w_out_final / (w_out + 1e-6), 0, 0], [0, h_out_final / (h_out + 1e-6), 0], [0, 0, 1]])

	H = np.matmul(H_resize, H)

	w_out, h_out, H_trans, min_x, max_x = calculate_raster(img, H)

	# print("w_out, h_out: ", w_out, h_out)

	H = np.matmul(H_trans, H)

	new_img = np.zeros((h_out, w_out, 3), dtype=np.uint8)

	return new_img, H


def two_step(img_points, img, img_affine_points):

	img_height, img_width, _ = img.shape

	straight_line_horizontal1 = np.cross(HC(img_points[0]), HC(img_points[1]))
	straight_line_vertical1 = np.cross(HC(img_points[1]), HC(img_points[2]))

	straight_line_horizontal2 = np.cross(HC(img_points[3]), HC(img_points[2]))
	straight_line_vertical2 = np.cross(HC(img_points[0]), HC(img_points[3]))

	vanishing_point1 = np.cross(straight_line_horizontal1, straight_line_horizontal2)
	vanishing_point2 = np.cross(straight_line_vertical1, straight_line_vertical2)

	vanishing_line = np.cross(vanishing_point1, vanishing_point2)
	vanishing_line = vanishing_line / (vanishing_line[2])


	Hp = np.array([[1, 0, 0], [0, 1, 0], [vanishing_line[0], vanishing_line[1], vanishing_line[2]]])


	new_img1 = apply_homography_with_inverse(img, Hp)


	straight_line_horizontal1 = np.transpose(np.linalg.inv(Hp)) @ straight_line_horizontal1
	straight_line_horizontal1 = straight_line_horizontal1 / straight_line_horizontal1[2]

	straight_line_horizontal2 = np.transpose(np.linalg.inv(Hp)) @ straight_line_horizontal2
	straight_line_horizontal2 = straight_line_horizontal2 / straight_line_horizontal2[2]

	straight_line_vertical1 = np.transpose(np.linalg.inv(Hp)) @ straight_line_vertical1
	straight_line_vertical1 = straight_line_vertical1 / straight_line_vertical1[2]

	straight_line_vertical2 = np.transpose(np.linalg.inv(Hp)) @ straight_line_vertical2
	straight_line_vertical2 = straight_line_vertical2 / straight_line_vertical2[2]


	L = np.array([[straight_line_horizontal1[0]*straight_line_vertical1[0], straight_line_horizontal1[0]*straight_line_vertical1[1] + straight_line_horizontal1[1]*straight_line_vertical1[0]], [straight_line_horizontal2[0]*straight_line_vertical2[0], straight_line_horizontal2[0]*straight_line_vertical2[1] + straight_line_horizontal2[1]*straight_line_vertical2[0]]]).reshape(2, 2)
	R = np.array([[-straight_line_horizontal1[1]*straight_line_vertical1[1]], [-straight_line_horizontal2[1]*straight_line_vertical2[1]]])


	L_inv = np.linalg.inv(L)

	result = np.squeeze(L_inv @ R)

	S = np.array([[result[0], result[1]], [result[1], 1]])

	U, D, Vt = np.linalg.svd(S)

	D_sqrt = np.diag(np.sqrt(D))

	A = U @ D_sqrt @ Vt

	
	Ha = np.array([[A[0, 0], A[0, 1], 0], [A[1, 0], A[1, 1], 0], [0, 0, 1]])

	H = Ha @ Hp

	new_img2 = apply_homography_with_inverse(img, H)



	new_img1 = cv2.cvtColor(new_img1, cv2.COLOR_BGR2RGB)
	new_img2 = cv2.cvtColor(new_img2, cv2.COLOR_BGR2RGB)

	plt.imshow(new_img1)
	plt.show()

	plt.imshow(new_img2)
	plt.show()


def one_step(img_1step_points, img):
	line11 = np.cross(HC(img_1step_points[0]), HC(img_1step_points[1]))
	line12 = np.cross(HC(img_1step_points[1]), HC(img_1step_points[2]))

	line21 = np.cross(HC(img_1step_points[3]), HC(img_1step_points[4]))
	line22 = np.cross(HC(img_1step_points[4]), HC(img_1step_points[5]))

	line31 = np.cross(HC(img_1step_points[6]), HC(img_1step_points[7]))
	line32 = np.cross(HC(img_1step_points[7]), HC(img_1step_points[8]))

	line41 = np.cross(HC(img_1step_points[9]), HC(img_1step_points[10]))
	line42 = np.cross(HC(img_1step_points[10]), HC(img_1step_points[11]))

	line51 = np.cross(HC(img_1step_points[12]), HC(img_1step_points[13]))
	line52 = np.cross(HC(img_1step_points[13]), HC(img_1step_points[14]))


	line_pairs = [[line11, line12], [line21, line22], [line31, line32], [line41, line42], [line51, line52]]


	L = []
	R = []

	for line_pair in line_pairs:
		l1 = line_pair[0][0]
		l2 = line_pair[0][1]
		l3 = line_pair[0][2]

		m1 = line_pair[1][0]
		m2 = line_pair[1][1]
		m3 = line_pair[1][2]

		L.append([l1 * m1, (l2*m1 + l1*m2) / 2, l2 * m2, (l3*m1 + l1*m3) / 2, (l3*m2 + l2*m3) / 2])
		R.append([-l3*m3])

	L = np.array(L)
	R = np.array(R)

	L_inv = np.linalg.inv(L)

	result = np.squeeze(L_inv @ R)

	a = result[0]
	b = result[1]
	c = result[2]
	d = result[3]
	e = result[4]

	AAt = np.array([[a, b/2], [b/2, c]])
	Av = np.array([[d/2], [e/2]])

	U, D, Vt = np.linalg.svd(AAt)

	A = U @ np.diag(np.sqrt(D)) @ Vt.T

	A_inv = np.linalg.inv(A)

	v = np.squeeze(A_inv @ Av)

	H = np.array([[A[0, 0], A[0, 1], 0], [A[1, 0], A[1, 1], 0], [v[0], v[1], 1]])

	H = H.T
	
	new_img = apply_homography_with_inverse(img, H)

	new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

	plt.imshow(new_img)
	plt.show()




state = '1step'

if(state == 'p2p'):
	# ## Finding Homographies
	H1 = find_homgoraphy(img1_points, transformed_img1_points)
	H1_test = find_homgoraphy(test_img1_points, transformed_test_img1_points)
	H2_test = find_homgoraphy(test_img2_points, transformed_test_img2_points)

	transformed_img1 = apply_homography_with_inverse(img1, H1)
	transformed_test_img1 = apply_homography_with_inverse(test_img1, H1_test)
	transformed_test_img2 = apply_homography_with_inverse(test_img2, H2_test)

	transformed_img1 = cv2.cvtColor(transformed_img1, cv2.COLOR_BGR2RGB)
	transformed_test_img1 = cv2.cvtColor(transformed_test_img1, cv2.COLOR_BGR2RGB)
	transformed_test_img2 = cv2.cvtColor(transformed_test_img2, cv2.COLOR_BGR2RGB)

	plt.imshow(transformed_img1)
	plt.show()

	plt.imshow(transformed_test_img1)
	plt.show()

	plt.imshow(transformed_test_img2)
	plt.show()

elif(state == '2step'):
	two_step(img1_points, img1, img1_2step_points)
	two_step(test_img1_points, test_img1, test_img1_2step_points)
	two_step(test_img2_points, test_img2, test_img2_2step_points)

elif(state == "1step"):
	one_step(img1_1step_points, img1)


