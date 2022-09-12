import numpy as np

class KNN:
    @classmethod
    def _knn(cls, x_train, x_test, k):
        # Exploded form of (a-b)^2
        distances = np.sum(x_train ** 2, axis=1)[:, np.newaxis] + np.sum(x_test ** 2, axis=1) - 2 * x_train@x_test.T
        closest_idxs = np.argsort(distances, axis=0)[:k]
        return closest_idxs

    @classmethod
    def knn(cls, X_train, X_test, y, k):
        closest_indices = cls._knn(X_train, X_test, k)
        predictions = np.take(y, closest_indices)
 
        predicted_labels = []
        for x_prediction in predictions.T:
            c = np.bincount(x_prediction)
            predicted_labels.append(np.argmax(c))

        return np.asarray(predicted_labels)

class Loss:
    @staticmethod
    def binary(truths, predictions) -> np.ndarray:
        return (predictions == truths).astype(int)

    @staticmethod
    def mse(y_hats, ys) -> float:
        return ((ys - y_hats)**2).mean(axis=0)

    @staticmethod
    def absolute(truths, predictions) -> np.ndarray:
        return 1


class Distance:
    @staticmethod
    def euclid_sq(x1, x2) -> np.ndarray:
        return np.sqrt(np.sum((x1 - x2)**2, axis=1))