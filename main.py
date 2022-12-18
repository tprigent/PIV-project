import sift_detect
import kp_matching
import cv2          # for debug only


if __name__ == "__main__":

    # Extract key points and descriptors from images
    kp_template, d_template = sift_detect.extract_kp_des('data/templateSNS.jpg', n_kpoints=200)
    kp_picture, d_picture = sift_detect.extract_kp_des('data/rgb0001.jpg', n_kpoints=200)

    # Find matches between descriptors
    matches = kp_matching.brute_force_matcher(d_template, d_picture)


    ## DEBUG ONLY: show matched key points
    image1 = cv2.imread('data/templateSNS.jpg')
    image2 = cv2.imread('data/rgb0001.jpg')
    matches = [cv2.DMatch(matches[i][0], matches[i][1], 0) for i in range(len(matches))]
    result = cv2.drawMatches(image1, kp_template, image2, kp_picture, matches, None)

    # Show the result
    cv2.imshow('Matched Keypoints', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
