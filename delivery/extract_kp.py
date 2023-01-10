import cv2
import os
import numpy as np
import scipy
from tqdm import tqdm


def extract_kp_des(image_path, n_keypoints):
    image = cv2.imread(image_path)
    image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(n_keypoints)
    key_points, descriptors = sift.detectAndCompute(image_g, None)

    return key_points, descriptors


if __name__ == "__main__":
    extensions = ['.jpg', '.jpeg', '.png']
    input_files = [f for f in os.listdir('input/') if any(f.endswith(ext) for ext in extensions)]
    template_files = [f for f in os.listdir('template/') if any(f.endswith(ext) for ext in extensions)]

    # getting keypoints and descriptors for pictures
    for picture_name in tqdm(input_files):
        kp, desc = extract_kp_des('input/' + picture_name, 200)
        kp_array = np.asarray([kp[i].pt for i in range(len(kp))])
        desc_array = np.asarray(desc, dtype=np.int32)
        scipy.io.savemat('input/'+picture_name+'.mat', {'p': kp_array, 'd': desc_array})

    # getting keypoints and descriptors for templates
    for picture_name in tqdm(template_files):
        kp, desc = extract_kp_des('template/' + picture_name, 200)
        kp_array = np.asarray([kp[i].pt for i in range(len(kp))])
        desc_array = np.asarray(desc, dtype=np.int32)
        scipy.io.savemat('template/' + picture_name + '.mat', {'p': kp_array, 'd': desc_array})