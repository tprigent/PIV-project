import numpy as np
import cv2


# Applies transform_im to every image in the frame_list with its corresponding H matrix
def transform_vid(H_list, frame_list):
    new_frame_list = np.empty_like(frame_list)
    for i, frame in enumerate(frame_list):
        new_frame = transform_im(H_list[i], frame)
        new_frame_list[i] = new_frame

    return new_frame_list


# Returns None if H is not invertible
def transform_im(H, frame, shape_template):

    # Checks if H is an homography
    # if H[2, 2] != 1:
    #     raise Exception("H(3,3) isn't equal to 1, the matrix is not an homography matrix")
    # Create the original image matrix
    frame = cv2.imread(frame)

    # Create variables that calculate the dimensions of the frames
    n_old, m_old, _ = np.shape(frame)
    n_new, m_new, _ = shape_template
    # n_new, m_new, x_new = tuple((H.dot(np.array([m_old, n_old, 1]))))
    # n_new, m_new = (int(n_new / x_new), int(m_new / x_new))

    # Create the new frame
    new_frame = np.zeros((n_new, m_new, 3)).astype(np.uint8)

    # Check if H is a singular matrix
    if np.linalg.det(H) != 0:
        H_inv = np.linalg.inv(H)
    else:
        return None

    # Calculate the values of the pixels of the new frame
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
                new_frame[i, j] = frame[x_old, y_old]

    return new_frame

    # test --------------------------

    # highest_x = 0
    # lowest_x = 0
    # highest_y = 0
    # lowest_y = 0
    #
    # for i in range(n_old-1):
    #     for j in range(m_old-1):
    #         homo_coord = np.array([i, j, 1])
    #         new_homo_coord = H.dot(homo_coord)
    #         x_new, y_new = int(new_homo_coord[0] / new_homo_coord[2]), int(new_homo_coord[1] / new_homo_coord[2])
    #         if x_new > highest_x:
    #             highest_x = x_new
    #         if x_new < lowest_x:
    #             lowest_x = x_new
    #         if y_new > highest_y:
    #             highest_y = y_new
    #         if y_new < lowest_y:
    #             lowest_y = y_new
    #
    # n_new = highest_x - lowest_x
    # m_new = highest_y - lowest_y
    # # Create the new frame
    # new_frame = np.zeros((n_new, m_new, 3)).astype(np.uint8)
    # for i in range(n_old):
    #     for j in range(m_old):
    #         homo_coord = np.array([i, j, 1])
    #         new_homo_coord = H.dot(homo_coord)
    #         x_new, y_new = int((new_homo_coord[0]) / new_homo_coord[2]), int((new_homo_coord[1]) / new_homo_coord[2])
    #         if 0 <= x_new - lowest_x < n_new and 0 <= y_new - lowest_y < m_new:
    #             new_frame[x_new - lowest_x, y_new - lowest_y] = frame[i, j]
    # A = np.array([[[0,0,0], [0,0,0]],
    #               [[0,0,0], [0,0,0]]])
    #
    # print(A.shape)
    # print(np.delete(A, np.where(A == [0, 0, 0], A)))
    # new_frame = np.delete(new_frame, np.where(new_frame == [0, 0, 0]))
    # return new_frame
