import os
import cv2
import scipy
import trans_func
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    # open template image
    template_frame = cv2.imread('template/templateSNS.jpg')

    files = os.listdir('data')
    num_input = int(len(files) / 2)

    for iteration_counter in tqdm(range(1, num_input)):
        input_name = 'data/rgb{}.jpg'.format(str(iteration_counter).zfill(4))
        input_frame = cv2.imread(input_name)

        H_mat_name = 'H_{}.mat'.format(str(iteration_counter).zfill(4))
        H_mat_packed = scipy.io.loadmat('results/'+H_mat_name)
        H = np.array(H_mat_packed['H'])

        new_frame = trans_func.transform_im(H, input_frame, template_frame)

        new_frame_name = 'tf{}.jpg'.format(str(iteration_counter).zfill(4))
        cv2.imwrite('results/'+new_frame_name, new_frame)

        iteration_counter += 1
