import numpy as np


def brute_force_matcher(descriptors1, descriptors2):
    n_desc1 = descriptors1.shape[0]
    n_desc2 = descriptors2.shape[0]

    correspondences = []

    # Descriptors of the 1st set
    for i in range(n_desc1):
        descriptor1 = descriptors1[i]

        best_match = None
        min_distance = float('inf')

        # Descriptors for the 2nd set
        for j in range(n_desc2):
            descriptor2 = descriptors2[j]

            # Calculate the distance between the descriptors
            distance = np.linalg.norm(descriptor1 - descriptor2)

            # Check if j is better candidate
            if distance < min_distance:
                best_match = j
                min_distance = distance

        # Update result list with best candidate
        correspondences.append((i, best_match))

    return correspondences
