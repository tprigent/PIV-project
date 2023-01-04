import numpy as np
import random


class HomographyModel:
    def fit(self, keypoints1, keypoints2):
        """
        Fit a homography matrix to the given keypoints.

        Parameters:
            keypoints1 - a set of 2D keypoints in the form (x, y)
            keypoints2 - a set of 2D keypoints in the form (x, y)

        Returns:
            H - a 3x3 homography matrix
        """
        # Convert the keypoints to homogeneous coordinates
        kpts1 = [keypoints1[i].pt for i in range(len(keypoints1))]
        kpts2 = [keypoints2[i].pt for i in range(len(keypoints2))]
        points1 = np.vstack((np.array(kpts1).T, np.ones(len(kpts1))))
        points2 = np.vstack((np.array(kpts2).T, np.ones(len(kpts1))))

        # Solve for the homography matrix using the least squares method
        A = []
        for i in range(points1.shape[1]):
            x, y, w = points1[:, i]
            u, v, _ = points2[:, i]
            A.append([x, y, w, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, w, -v * x, -v * y, -v])
        A = np.asarray(A)
        U, S, Vh = np.linalg.svd(A)
        L = Vh[-1, :] / Vh[-1, -1]
        H = L.reshape(3, 3)

        return H

    def distance(self, H, keypoint1, keypoint2):
        """
        Calculate the distance between a keypoint and the model.

        Parameters:
            H - a 3x3 homography matrix
            keypoint1 - a 2D keypoint in the form (x, y)
            keypoint2 - a 2D keypoint in the form (x, y)

        Returns:
            d - the distance between the keypoint and the model
        """
        # Transform the keypoint using the homography matrix
        point1 = np.asarray([keypoint1.pt[0], keypoint1.pt[1], 1])
        point2 = np.asarray([keypoint2.pt[0], keypoint2.pt[1], 1])
        point2_transformed = np.dot(H, point1)
        point2_transformed /= point2_transformed[2]

        # Calculate the distance between the transformed keypoint and the original keypoint
        d = np.linalg.norm(point2 - point2_transformed)

        return d

    def get_error(self, H, keypoints1, keypoints2):
        """
        Calculate the total error between the model and the data.

        Parameters:
            H - a 3x3 homography matrix
            keypoints1 - a set of 2D keypoints in the form (x, y)
            keypoints2 - a set of 2D keypoints in the form (x, y)

        Returns:
            err - the total error between the model and the data
        """
        err = 0
        for i in range(len(keypoints1)):
            err += self.distance(H, keypoints1[i], keypoints2[i])
        return err


def ransac(keypoints1, keypoints2, model, n, k, t, d, debug=True):
    """
    Fit a model to data using the RANSAC algorithm.

    Parameters:
        keypoints1 - a set of 2D keypoints in the form (x, y)
        keypoints2 - a set of 2D keypoints in the form (x, y)
        model - a model that can be fitted to data
        n - the minimum number of data required to fit the model
        k - the maximum number of iterations allowed in the algorithm
        t - a threshold value for determining when a datum fits a model
        d - the number of close data values required to assert that a model fits well to data
        debug - an optional flag to indicate debugging mode

    Returns:
        bestfit - model parameters which best fit the data (or None if no good model is found)
    """
    bestfit = None
    besterr = np.inf
    bestinliers = None
    for i in range(k):
        indexes = random.sample(range(len(keypoints1)), n)
        maybeinliers1 = [keypoints1[i] for i in indexes]
        maybeinliers2 = [keypoints2[i] for i in indexes]
        # maybeinliers1, maybeinliers2 = random.sample(keypoints1, n), random.sample(keypoints2, n)
        maybemodel = model.fit(maybeinliers1, maybeinliers2)
        alsoinliers1, alsoinliers2 = [], []
        inliers_index = []
        for j in range(len(keypoints1)):
            if model.distance(maybemodel, keypoints1[j], keypoints2[j]) < t:
                alsoinliers1.append(keypoints1[j])
                alsoinliers2.append(keypoints2[j])
                inliers_index.append(j)
        if len(alsoinliers1) > d:
            bettermodel = model.fit(alsoinliers1, alsoinliers2)
            thiserr = model.get_error(bettermodel, alsoinliers1, alsoinliers2)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                bestinliers = inliers_index
    if debug:
        print(f"best fit: {bestfit} with error {besterr}")

    return bestfit, bestinliers
