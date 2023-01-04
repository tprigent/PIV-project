import cv2
import numpy as np


def extract_kp_des(image_name, n_kpoints):
    # Load the two images and convert them to grayscale
    image = cv2.imread(image_name)

    # Convert images to array
    im_array = np.array(image)
    im_array = im_array.astype(np.uint8)

    # apply SIFT to both images to get key points and descriptors
    sift = cv2.SIFT_create(nfeatures=n_kpoints, contrastThreshold=0.01, edgeThreshold=0.01, sigma=3)
    key_points, descriptors = sift.detectAndCompute(im_array, None)

    return key_points, descriptors




