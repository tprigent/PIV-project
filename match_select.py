from scipy.ndimage.interpolation import shift
import numpy as np
import random
from tqdm import tqdm
import cv2


def ransac(keypoints1, keypoints2, n_iter, n_data, th, n_valid):
    # Convert the keypoints to points
    kpts1 = [keypoints1[i].pt for i in range(len(keypoints1))]
    kpts2 = [keypoints2[i].pt for i in range(len(keypoints2))]

    # memory
    best_fit = None
    min_error = np.inf
    best_inliers = None

    # iterating n times
    for i in tqdm(range(n_iter)):
        # taking n random keypoints in each side
        sample_indexes = random.sample(range(len(keypoints1)), n_data)
        candidates1 = [kpts1[i] for i in sample_indexes]
        candidates2 = [kpts2[i] for i in sample_indexes]

        # determine fit function
        model = compute_homography(candidates1, candidates2)

        # find inliers
        #mse = measure_accuracy(model, candidates1, candidates2)
        #print('mse = {}'.format(mse))
        inliers1, inliers2, inlier_indexes = [], [], []
        for j in range(len(kpts1)):
            if distance(model, kpts1[j], kpts2[j]) < th:
                inliers1.append(kpts1[j])
                inliers2.append(kpts2[j])
                inlier_indexes.append(j)

        # detect inlier relevance
        if len(inliers1) > n_valid:
            best_model = compute_homography(points1=inliers1, points2=inliers2)
            curr_error = compute_error(best_model, points1=inliers1, points2=inliers2)

            # select best candidate
            if curr_error < min_error:
                best_fit = best_model
                min_error = curr_error
                best_inliers = inlier_indexes

    print('Min error = {}'.format(min_error))

    return best_fit, best_inliers


def compute_homography(points1, points2):  # fit function
    n = len(points1)

    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = points1[i]
        u, v = points2[i]
        A[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y, -u]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, -v]

    _, _, V = np.linalg.svd(A)
    H = V[-1, :]
    H = H.reshape((3, 3))

    return H


def distance(homography, point1, point2):
    # convert to homogeneous coord.
    x, y = point1[0], point1[1]
    u, v = point2[0], point2[1]
    h_point1 = np.asarray([int(x), int(y), 1])
    h_point2 = np.asarray([int(u), int(v), 1])

    # apply homography to point2
    new_point2 = np.dot(homography, h_point1)

    # get new point standard expression
    if new_point2[2] != 0:
        new_point2 /= new_point2[2]

    # compute distance between reference point and output point
    d = np.linalg.norm(h_point2 - new_point2)

    return d


def compute_error(homography, points1, points2):
    err = 0
    for i in range(len(points1)):
        err += distance(homography, points1[i], points2[i])

    return err

