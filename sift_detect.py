import cv2


def extract_kp_des(image, n_keypoints):
    image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(n_keypoints)
    key_points, descriptors = sift.detectAndCompute(image_g, None)

    return key_points, descriptors




