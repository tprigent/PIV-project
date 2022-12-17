import sift_detect


if __name__ == "__main__":

    # Extract key points and descriptors from images
    kp_template, d_template = sift_detect.extract_kp_des('data/templateSNS.jpg', n_kpoints=200)
    kp_picture, d_picture = sift_detect.extract_kp_des('data/rgb0001.jpg', n_kpoints=200)
