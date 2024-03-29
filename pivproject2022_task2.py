import sys
import os
import scipy
import render
import numpy as np

import sift_detect
import panorama_maker

def compute_homography_for_dataset(ref_number, input_dir, output_dir):
    given_keypoints = 1
    
    first_name = 'rgbsift_{}'.format(str(ref_number).zfill(4))

    # unpacking template data

    kp_template, desc_template = get_keypoints(input_dir+first_name, given_keypoints)

    files = os.listdir(input_dir)
    num_input = int(len(files)/2)
    inliers_index = None

    start = 1
    end = num_input+1

    for iteration_counter in range(start, end):
        if(iteration_counter == ref_number):
            continue

        # unpacking input data
        matinput_name = 'rgbsift_{}'.format(str(iteration_counter).zfill(4))
        kp_input, desc_input = get_keypoints(input_dir + matinput_name, given_keypoints)

        # raw matching descriptors
        correspondences = descriptor_matcher(descriptors1=desc_input, descriptors2=desc_template)
        match_template = [correspondences[i][1] for i in range(len(correspondences))]
        match_input = [correspondences[i][0] for i in range(len(correspondences))]

        # get corresponding matched keypoints
        kp_t_match = [kp_template[int(i)] for i in match_template]
        kp_i_match = [kp_input[int(i)] for i in match_input]

        # select good matches and compute homography
        H = None
        tolerance = 0
        while H is None:
            if given_keypoints:
                H, inliers_index = ransac(kp_t_match, kp_i_match, n_iter=200, n_data=4, th=5+tolerance, n_valid=20)
            else:
                H, inliers_index = ransac(kp_t_match, kp_i_match, n_iter=200, n_data=4, th=2+tolerance, n_valid=30)
            tolerance += 1

        H_mat_name = 'H_{}.mat'.format(str(iteration_counter).zfill(4))
        scipy.io.savemat(output_dir + H_mat_name, {'H': H})

        matches = [(int(match_template[i]), int(match_input[i]), 0) for i in inliers_index]
        render.draw_matches(kp_template, kp_input, matches, iteration_counter)


def get_keypoints(image_name, given_keypoints=1):

    if given_keypoints:
        matrix = scipy.io.loadmat(image_name + '.mat')
        kp = np.array(matrix['p']).T
        desc = np.array(matrix['d']).T
    else:
        kp, desc = sift_detect.extract_kp_des(image_name + '.jpg', 200)
        kp = np.asarray(kp)
        desc = np.asarray(desc)
        for i in range(len(kp)):
            kp[i] = kp[i].pt

    return kp, desc


def descriptor_matcher(descriptors1, descriptors2):

    tree = scipy.spatial.cKDTree(descriptors1)
    distances, all_matches = tree.query(descriptors2)

    matches = []

    for i in range(len(distances)):
        if distances[i] < 180:
            matches.append([all_matches[i], i])

    return np.asarray(matches)


def ransac(kpts1, kpts2, n_iter, n_data, th, n_valid):
    # memory
    best_fit = None
    min_error = np.inf
    best_inliers = None

    # iterating n times
    for i in range(n_iter):
        # taking n random keypoints in each side
        sample_indexes = np.random.choice(range(len(kpts1)), n_data, replace=False)
        candidates1 = [kpts1[i] for i in sample_indexes]
        candidates2 = [kpts2[i] for i in sample_indexes]

        # determine fit function
        model = compute_homography(candidates1, candidates2)

        # find inliers
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
            curr_error/len(inliers1)

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
        x, y = points2[i]
        u, v = points1[i]
        A[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y, -u]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, -v]

    _, _, V = np.linalg.svd(A)
    H = V[-1, :]
    H = H.reshape((3, 3))
    H = H / H[-1, -1]
    return H


def distance(homography, point1, point2):
    # convert to homogeneous coord.
    x, y = point1[0], point1[1]
    u, v = point2[0], point2[1]
    h_point1 = np.asarray([int(x), int(y), 1])
    h_point2 = np.asarray([int(u), int(v), 1])

    # apply homography to point2
    new_point2 = np.dot(homography, h_point2)

    # get new point standard expression
    if new_point2[2] != 0:
        new_point2[0] = new_point2[0]/new_point2[2]
        new_point2[1] = new_point2[1]/new_point2[2]
        new_point2[2] = 1

    #new_point2 = new_point2[:2]
    #h_point1 = h_point1[:2]

    # compute distance between reference point and output point
    d = np.linalg.norm(h_point1 - new_point2)

    return d


def compute_error(homography, points1, points2):
    err = 0
    for i in range(len(points1)):
        err += distance(homography, points1[i], points2[i])

    return err


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 3:
        ref_number = args[1]
        input_dir = args[2]
        output_dir = args[3]
    else:
        print('NO ARGUMENTS FOUND USING STANDARD PATH')
        ref_number = 1
        input_dir = 'data/'
        output_dir = 'results/'

    compute_homography_for_dataset(ref_number, input_dir, output_dir)
    panorama_maker.makePanorama(ref_number,input_dir,output_dir)




