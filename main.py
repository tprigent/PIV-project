import numpy as np

import cleaning
import ransac
import sift_detect
import kp_matching
import cv2  # for debug only
from Trans_func import transform_im

if __name__ == "__main__":
    # Extract key points and descriptors from images
    kp_template, d_template = sift_detect.extract_kp_des('data/templateSNS.jpg', n_kpoints=500)
    kp_picture, d_picture = sift_detect.extract_kp_des('data/rgb0001.jpg', n_kpoints=500)

    # Find matches between descriptors
    match_template, match_picture = kp_matching.distance_matcher(d_template, d_picture)
    kp_t_match = [kp_template[int(i)] for i in match_template]
    kp_p_match = [kp_picture[int(i)] for i in match_picture]

    # Use RANSAC to select the best matches
    homography_model = ransac.HomographyModel()
    H, inliers_index = ransac.ransac(kp_t_match, kp_p_match, homography_model, n=4, k=2000, t=2, d=50)

#    matches_matrix = np.array([i for i in matches])
#    new_matches = matches_matrix[filter]

    ## DEBUG ONLY: show matched key points

    image1 = cv2.imread('data/templateSNS.jpg')
    image2 = cv2.imread('data/rgb0001.jpg')
    # new_matches = [cv2.DMatch(inliers_template[i].pt, inliers_picture[i].pt, 0) for i in range(len(inliers_template))]
    matches = [cv2.DMatch(int(match_template[i]), int(match_picture[i]), 0) for i in inliers_index]

    result = cv2.drawMatches(image1, kp_t_match, image2, kp_p_match, matches, None)

    # Show the result

    # cv2.imshow('Matched Keypoints', result)
    cv2.imwrite('results/result_task_1.png', result)

    new_frame = transform_im(H, 'data/rgb0001.jpg')
    cv2.imwrite('results/result_after_im_transform.png', new_frame)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
