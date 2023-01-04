import numpy as np


def distance_matcher(descriptors1, descriptors2):
    index1 = np.zeros(len(descriptors1))
    index2 = np.zeros(len(descriptors1))
    cnt_desc1 = 0
    for template_descriptor in descriptors1:
        distances = np.linalg.norm(template_descriptor - descriptors2, axis=1)
        best_match = np.argmin(distances)
        index1[cnt_desc1] = cnt_desc1
        index2[cnt_desc1] = best_match
        cnt_desc1 += 1
    return index1, index2
