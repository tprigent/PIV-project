import os
import cv2
import time
import scipy
import trans_func
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    # open template image
    template_frame = cv2.imread('template/templateSNS.jpg')

    files = os.listdir('data')
    num_input = int(len(files) / 2)

    start = 16
    end = num_input

    for iteration_counter in tqdm(range(start, end)):
        input_name = 'data/rgb{}.jpg'.format(str(iteration_counter).zfill(4))
        input_frame = cv2.imread(input_name)

        H_mat_name = 'H_{}.mat'.format(str(iteration_counter).zfill(4))
        H_mat_packed = scipy.io.loadmat('results/'+H_mat_name)
        H = np.array(H_mat_packed['H'])

        new_frame = trans_func.transform_im(H, input_frame, template_frame)

        new_frame_name = 'tf{}.jpg'.format(str(iteration_counter).zfill(4))
        cv2.imwrite('results/'+new_frame_name, new_frame)


def draw_matches(points1, points2, matches, iteration):
    image1 = cv2.imread('template/templateSNS.jpg')
    im2_name = 'data/rgb{}.jpg'.format(str(iteration).zfill(4))
    image2 = cv2.imread(im2_name)



    # Create a new image to draw the lines on
    pad_size = image1.shape[0] - image2.shape[0]
    image2 = np.pad(image2, [(0, pad_size), (0, 0), (0, 0)], mode="constant")
    matches_image = np.concatenate((image1, image2), axis=1)

    # Draw the lines between the points
    for i in range(len(matches)):
        x, y = points1[matches[i][0]]
        u, v = points2[matches[i][1]]
        u = u + image1.shape[1]

        cv2.line(matches_image, (int(x), int(y)), (int(u), int(v)), (255, 0, 0), 1)
        cv2.circle(matches_image, (int(x), int(y)), 1, (0, 255, 0), 3)
        cv2.circle(matches_image, (int(u), int(v)), 1, (0, 0, 255), 3)

    # Return the image with the lines drawn on it
    match_name = 'matching_{}.jpg'.format(str(iteration).zfill(4))
    cv2.imwrite('results/'+match_name, matches_image)
