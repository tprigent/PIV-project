import numpy as np


class HomographyModel:
    def fit(self, samples):
        """
        Fit a homography matrix to a set of samples.
        
        Parameters
        ----------
        samples: ndarray
            The input samples, an Nx4 array where each row represents a pair of corresponding points in the form (x1, y1, x2, y2).
        """
        # Create the matrix A for the linear system Ah = 0
        A = np.zeros((len(samples) * 2, 9))
        for i, (x1, y1, x2, y2) in enumerate(samples):
            A[i * 2] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
            A[i * 2 + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]

        # Compute the SVD of A
        U, S, Vt = np.linalg.svd(A)
        # The homography matrix is the last column of V so the last row of Vt
        self.homography = Vt[-1].reshape(3, 3)
        self.homography = self.homography/self.homography[2, 2]

    def predict(self, data):
        """
        Predict the output values for a set of data points.
        
        Parameters
        ----------
        data: ndarray
            The input data, an Nx2 array where each row represents a point in the form (x, y).
        
        Returns
        -------
        ndarray
            The predicted output values, an Nx2 array where each row represents a point in the form (x', y').
        """
        # Add a third coordinate to the data points
        data_homogeneous = np.hstack((data, np.ones((len(data), 1))))

        # Transform the data points using the homography matrix
        transformed = np.dot(self.homography, data_homogeneous.T).T

        # Normalize the transformed points
        transformed[:, :2] /= transformed[:, 2:]

        return transformed[:, :2]


def ransac(data, model_class, min_samples, residual_threshold, max_trials):
    """
    Fit a model to data using the RANSAC algorithm.
    
    Parameters
    ----------
    data: ndarray
        The input data in the form (x1, y1, x2, y2) for each index.
    model_class: type
        The class of the model to fit to the data.
        The model should have a fit method and a predict method.
    min_samples: int
        The minimum number of samples required to fit the model.
    residual_threshold: float
        The maximum distance between a sample and the model that is considered
        a good fit.
    max_trials: int
        The maximum number of iterations to perform before giving up.
    
    Returns
    -------
    model: object
        The best fit model.
    inliers: ndarray
        The indices of the samples that fit the model well.
    """
    best_model = None
    best_inliers = None

    # Perform max_trials iterations
    for trial in range(max_trials):
        # Randomly select min_samples points from the data
        indices = np.random.choice(len(data), min_samples, replace=False)
        samples = data[indices]
        # Fit a model to the samples
        model = model_class()
        model.fit(samples)
        data_picture = data[:, 2:]
        data_template = data[:, :2]

        # Use the model to predict the remaining data points
        predictions = model.predict(data_picture)

        # Compute the residuals between the predictions and the actual data points
        residuals = predictions - data_template
        distance_vector = np.array([np.linalg.norm(i) for i in residuals])

        # Count the number of inliers (samples that have a small residual)
        inliers = np.sum(distance_vector < residual_threshold)

        # If the model has more inliers than the current best model, update the best model
        if best_inliers is None or inliers > best_inliers:
            best_model = model
            best_distance_vector = distance_vector
            best_inliers = inliers
    # Return the best fit model and the inliers

    print(best_inliers)
    print(model.homography)
    filter = np.where(best_distance_vector < residual_threshold, True, False)
    return best_model.homography, filter
