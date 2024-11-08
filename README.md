This repository contains my project submissions for ECE 661 - Computer Vision
- hw1 is a python script that uses homogeneous coordinates to determine if a given aiming angle in a space-invaders-type game will result in a hit with the triangle-shaped spaceship
- hw2 covers estimating homographies between images and applying them to each other to perform simple transformations
- hw3 is about removing projective and affine distortions from various images using 2-step and 1-step techniques
- hw4 is a python script that uses a variety of techniques to extract interest points from images and find correspondent points between pairs of images of the same object. My implementation uses a Harris corner detector, the SIFT algorithm, and the SuperPoint/SuperGlue framework
- hw5 expands on hw4 and utilizes the point correspondency matching techniques to find the interest points, then feeds them into RANSAC to perform outlier rejection, and finally estimates the homography from the inliers and refines the homography to result in a panorama image made of 5 individual images as input
