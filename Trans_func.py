import numpy as np
import cv2


# Applies transform_im to every image in the transpose_frame_list with its corresponding H matrix
def transform_vid(H_list, transpose_frame_list):
    new_frame_list = np.empty_like(transpose_frame_list)
    for i, transpose_frame in enumerate(transpose_frame_list):
        new_frame = transform_im(H_list[i], transpose_frame)
        new_frame_list[i] = new_frame

    return new_frame_list


# Returns None if H is not invertible
def transform_im(H, frame_path):

    # Checks if H is an homography
    # if H[2, 2] != 1:
    #     raise Exception("H(3,3) isn't equal to 1, the matrix is not an homography matrix")
    # Create the original image matrix

    frame = cv2.imread(frame_path)
    
    transpose_frame = np.zeros((frame.shape[1], frame.shape[0], frame.shape[2])).astype(np.uint8)

    for i, line in enumerate(frame):
        transpose_frame[:, i] = line

    # Create variables that calculate the dimensions of the transpose_frames
    n_old, m_old, _ = np.shape(transpose_frame)
    n_new, m_new, x_new = tuple((H.dot(np.array([m_old, n_old, 1]))))
    n_new, m_new = (int(n_new / x_new), int(m_new / x_new))

    # Create the new transpose_frame
    new_frame = np.zeros((n_new, m_new, 3)).astype(np.uint8)

    # Check if H is a singular matrix
    if np.linalg.det(H) != 0:
        H_inv = np.linalg.inv(H)
    else:
        return None

    # Calculate the values of the pixels of the new transpose_frame
    for i, line in enumerate(new_frame):
        for j, pixel in enumerate(line):
            # Converts from cartesian to homogenous coordinates
            homo_coord = np.array([i, j, 1])

            # Retrieves the old coordinates to determine the pixel value
            old_homo_coord = H_inv.dot(homo_coord)

            # Convert back to cartesian
            x_old, y_old = int(old_homo_coord[0] / old_homo_coord[2]), int(old_homo_coord[1] / old_homo_coord[2])

            # Set the pixel value
            if 0 <= x_old < n_old and 0 <= y_old < m_old:
                new_frame[i, j] = transpose_frame[x_old, y_old]

    inverse_transpose_frame = np.zeros((new_frame.shape[1], new_frame.shape[0], new_frame.shape[2])).astype(np.uint8)

    for i, line in enumerate(new_frame):
        inverse_transpose_frame[:, i] = line

    return inverse_transpose_frame
