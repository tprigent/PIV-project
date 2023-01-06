import glob

import numpy as np
import os
import ransac
import sift_detect
import kp_matching
import match_select
import cv2  # for debug only
from Trans_func import transform_im

if __name__ == "__main__":

    files = glob.glob('results/*')
    for f in files:
        os.remove(f)

    template = cv2.imread('data/templateSNS.jpg')
    picture = cv2.imread('data/rgb0001.jpg')

    # Extract key points and descriptors from images
    n_keypoints = 200
    kp_template, d_template = sift_detect.extract_kp_des(template, n_keypoints)
    kp_picture, d_picture = sift_detect.extract_kp_des(picture, n_keypoints)

    # Find matches between descriptors
    correspondences = kp_matching.distance_matcher(d_template, d_picture)
    match_template = [correspondences[i] for i in range(len(correspondences))]
    match_picture = [i for i in range(len(correspondences))]

    kp_t_match = [kp_template[int(i)] for i in match_template]
    kp_p_match = [kp_picture[int(i)] for i in match_picture]

    # Use RANSAC to select the best matches
    homography_model = ransac.HomographyModel()
    # H, inliers_index = ransac.ransac(kp_t_match, kp_p_match, homography_model, n_data=4, n_iter=2000, th=20, n_validation=20)
    # H, inliers_index = match_select.ransac(kp_t_match, kp_p_match, n_iter=1000, n_data=4, th=2, n_valid=8)
    # H, inliers_index = match_select.ransac(kp_t_match, kp_p_match, n_iter=1000, n_data=4, th=2, n_valid=20)
    H, inliers_index = match_select.ransac(kp_t_match, kp_p_match, n_iter=1000, n_data=4, th=2, n_valid=20)


    ## DEBUG ONLY: show matched key points

    raw_matches = [cv2.DMatch(int(match_template[i]), int(match_picture[i]), 0) for i in range(len(match_template))]
    raw_result = cv2.drawMatches(template, kp_template, picture, kp_picture, raw_matches, None)
    cv2.imwrite('results/matches_raw.jpg', raw_result)

    matches = [cv2.DMatch(int(match_template[i]), int(match_picture[i]), 0) for i in inliers_index]
    result = cv2.drawMatches(template, kp_template, picture, kp_picture, matches, None)
    cv2.imwrite('results/matches.jpg', result)

    new_frame = transform_im(H, 'data/rgb0001.jpg')
    opencv_frame = cv2.warpPerspective(picture, H, template.shape[:2])

    cv2.imwrite('results/im_transform.jpg', new_frame)
    cv2.imwrite('results/im_opencv.jpg', opencv_frame)


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
