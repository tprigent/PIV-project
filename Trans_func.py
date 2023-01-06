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
def transform_im(H, frame):
    # Checks if H is an homography
    # if H[2, 2] != 1:
    #     raise Exception("H(3,3) isn't equal to 1, the matrix is not an homography matrix")
    # Create the original image matrix
    frame = cv2.imread(frame)


    # Create variables that calculate the dimensions of the frames
    n_old, m_old, _ = np.shape(frame)
    n_new, m_new, x_new = tuple((H@np.array([n_old, m_old, 1])))
    n_new, m_new = (int(n_new / x_new), int(m_new / x_new))

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
            old_homo_coord = H_inv@homo_coord

            # Convert back to cartesian
            x_old, y_old = int(old_homo_coord[0] / old_homo_coord[2]), int(old_homo_coord[1] / old_homo_coord[2])

            # Set the pixel value
            if 0 <= x_old < n_old and 0 <= y_old < m_old:
                new_frame[i, j] = frame[x_old, y_old]

    return new_frame


# H = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
# img = "data/rgb0001.jpg"
# original_img = cv2.imread(img)
# transformed = transform_im(H, img)
# grey = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
# print(np.array_equal(original_img, transformed))
# #open_cv = cv2.warpPerspective(frame, H, (1000, 1000))
#
# cv2.imshow("original", original_img)
# cv2.imshow("transed", transformed)
# #cv2.imshow("cv", open_cv )
#
# cv2.waitKey()
