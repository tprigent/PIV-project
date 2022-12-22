import numpy as np
import ransac
import sift_detect
import kp_matching
import cv2  # for debug only
from Trans_func import transform_im

if __name__ == "__main__":

    # Extract key points and descriptors from images
    kp_template, d_template = sift_detect.extract_kp_des('data/templateSNS.jpg', n_kpoints=200)
    kp_picture, d_picture = sift_detect.extract_kp_des('data/rgb0001.jpg', n_kpoints=200)

    # Find matches between descriptors
    matches = kp_matching.brute_force_matcher(d_template, d_picture)
    samples = np.array([(kp_template[i].pt + kp_picture[j].pt) for (i, j) in matches])
    model = ransac.HomographyModel
    max_trials = len(samples)*2
    H, filter = ransac.ransac(samples, model, 4, 300, max_trials)
    matches_matrix = np.array([i for i in matches])
    new_matches = matches_matrix[filter]

    ## DEBUG ONLY: show matched key points

    image1 = cv2.imread('data/templateSNS.jpg')
    image2 = cv2.imread('data/rgb0001.jpg')
    new_matches = [cv2.DMatch(new_matches[i][0], new_matches[i][1], 0) for i in range(len(new_matches))]
    result = cv2.drawMatches(image1, kp_template, image2, kp_picture, new_matches, None)

    # Show the result

    # cv2.imshow('Matched Keypoints', result)
    cv2.imwrite('results/result_task_1.png', result)

    # new_frame = transform_im(H, 'data/rgb0001.jpg')
    # cv2.imwrite('results/result_after_im_transform.png', new_frame)


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
