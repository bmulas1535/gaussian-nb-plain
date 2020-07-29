import numpy as np

class NaiveBayes:
    """Naive Bayes Classifier
    """

    def __init__(self):
        """This doesn't really do anything yet. 
        Creates instance of the object in memory.
        """
        return None

    def fit(self, X, y):
        """Fit the classifier on training data."""
        # Reduce shape into observations by features
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Create placeholder matrices for mean and variance
        self._mu = np.zeros((n_classes, n_features)).astype(float)
        self._sigma = np.zeros((n_classes, n_features)).astype(float)

        # Create placeholder matrix for prior probability (For each class)
        self.priors = np.zeros(n_classes).astype(float)

        for i, c in enumerate(self.classes):
            # Iterate over classes with index
            X_class = X[c==y]
            # assign shape values
            _observations, _features = X_class.shape

            # update class label mean and variance
            self._mu[i, :] = [
                X_class[:, idx].sum() / _observations for idx in range(_features)
            ]
            self._sigma[i, :] = [
                sum(x) / (_observations - 1) for x in [
                    x**2 for x in [
                        X_class[:, idx] - self._mu[i, idx] for idx in range(
                            _features
                        )
                    ]
                ]
            ]
            # Baseline probability is calculated as the ratio of observed class
            # labels out of the entire sample data
            self.priors[i] = _observations / n_samples


    def predict(self, X):
        """Predict classes for given feature matrix."""
        # make a list of predictions for each observation in the feature matrix
        y_pred = [self._single_pred(x) for x in X]
        return y_pred

    def _gdf(self, idx, x):
        """Gaussian distribution function."""
        _var = self._sigma[idx]
        _mean = self._mu[idx]
        numerator = np.exp(-(x - _mean)**2 / (2 * _var))
        denominator = (2 * np.pi * _var)**0.5
        return numerator / denominator

    def _single_pred(self, x):
        """Get prediction for single entry."""
        post = list()

        for idx, _class in enumerate(self.classes):
            _prior = self.priors[idx]
            _cc = np.prod(self._gdf(idx, x))
            _post = _prior * _cc
            post.append(_post)
        return self.classes[np.argmax(post)]

    def accuracy_score(self, X, y):
        """Do predictions on X and score on given true values."""
        y_pred = self.predict(X)
        
        accuracy = sum(y_pred==y) / len(y)
        return accuracy
