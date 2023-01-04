import numpy as np
from scipy.spatial import cKDTree


def distance_matcher(descriptors1, descriptors2):
    tree = cKDTree(descriptors1)
    _, matches = tree.query(descriptors2)
    correspondences = []

    for i in range(len(matches)):
        correspondences.append((i, matches))

    return correspondences
