import glob

import numpy as np
import os
import ransac
import sift_detect
import kp_matching
import cv2  # for debug only
from Trans_func import transform_im

if __name__ == "__main__":

    files = glob.glob('results/*')
    for f in files:
        os.remove(f)

    template = cv2.imread('data/templateSNS.jpg')
    picture = cv2.imread('data/rgb0001.jpg')

    # Extract key points and descriptors from images
    kp_template, d_template = sift_detect.extract_kp_des(template)
    kp_picture, d_picture = sift_detect.extract_kp_des(picture)

    # Find matches between descriptors
    correspondences = kp_matching.distance_matcher(d_template, d_picture)
    match_template = [correspondences[i] for i in range(len(correspondences))]
    match_picture = [i for i in range(len(correspondences))]

    kp_t_match = [kp_template[int(i)] for i in match_template]
    kp_p_match = [kp_picture[int(i)] for i in match_picture]

    # Use RANSAC to select the best matches
    homography_model = ransac.HomographyModel()
    H, inliers_index = ransac.ransac(kp_t_match, kp_p_match, homography_model, n_data=8, n_iter=2000, th=1, n_validation=10)

#    matches_matrix = np.array([i for i in matches])
#    new_matches = matches_matrix[filter]

    ## DEBUG ONLY: show matched key points

    matches_d = [cv2.DMatch(int(match_template[i]), int(match_picture[i]), 0) for i in range(len(match_template))]
    result = cv2.drawMatches(template, kp_template, picture, kp_picture, matches_d, None)
    cv2.imwrite('results/matchesKNN.jpg', result)

    new_frame = transform_im(H, 'data/rgb0001.jpg')
    cv2.imwrite('results/result_after_im_transform.png', new_frame)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
