# classifiers.py

import math

# Define a base class for ML classifiers so we can reuse the predict & report methods
class MLClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def predict_single(self, input_features):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def predict(self, X):
        return [self.predict_single(features) for features in X]

    def classification_report(self, y_true, y_pred):
        unique_labels = set(y_true)
        report = {}
        for label in unique_labels:
            tp = sum(1 for i in range(len(y_true)) if y_true[i] == label and y_pred[i] == label)
            fp = sum(1 for i in range(len(y_true)) if y_true[i] != label and y_pred[i] == label)
            fn = sum(1 for i in range(len(y_true)) if y_true[i] == label and y_pred[i] != label)
            tn = sum(1 for i in range(len(y_true)) if y_true[i] != label and y_pred[i] != label)

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) != 0 else 0

            report[label] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "accuracy": accuracy
            }

        return report

    def overall_metrics(self, report):
        accuracy = sum(report[label]['accuracy'] for label in report) / len(report)
        precision = sum(report[label]['precision'] for label in report) / len(report)
        recall = sum(report[label]['recall'] for label in report) / len(report)
        f1_score = sum(report[label]['f1-score'] for label in report) / len(report)
        return accuracy, precision, recall, f1_score

# Naive Bayes classifier
class NaiveBayesClassifier(MLClassifier):
    def __init__(self):
        # Dicts for mean, stdev, and class probabilities
        self.means = {}
        self.stds = {}
        self.class_probabilities = {}

    def fit(self, x, y):
        # Train the classifier by calculating the class probabilities
        # and the means and standard deviations for each feature
        self._calculate_class_probabilities(y)
        self._calculate_means_stds(x, y)

    def _calculate_class_probabilities(self, y):
        # Calculate the probability of each class based on label frequency
        class_counts = {label: y.count(label) for label in set(y)}
        total_count = len(y)
        self.class_probabilities = {label: count / total_count for label, count in class_counts.items()}

    def _calculate_means_stds(self, x, y):
        # Calculate the mean and standard deviation for each class and each feature
        for label in self.class_probabilities:
            # Extract features for instances of the current class
            label_features = [x[i] for i in range(len(x)) if y[i] == label]
            # Calculate mean and standard deviation for each feature
            self.means[label] = [sum(f) / len(f) for f in zip(*label_features)]
            self.stds[label] = [math.sqrt(sum([(x - mean)**2 for x in f]) / len(f)) for mean, f in zip(self.means[label], zip(*label_features))]

    def predict_single(self, input_features):
        # Predict the class of a single feature set
        probabilities = {}
        for label, _ in self.means.items():
            # Start with the prior probability of the class
            probabilities[label] = self.class_probabilities[label]
            # Multiply by the probability of each feature
            for i, feature in enumerate(input_features):
                probabilities[label] *= self._calculate_probability(feature, self.means[label][i], self.stds[label][i])
        # Return the class with the highest probability
        return max(probabilities, key=probabilities.get)

    def _calculate_probability(self, x, mean, std):
        # Calculate the probability of a feature value with a Gaussian distribution
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
        return (1 / (math.sqrt(2*math.pi) * std)) * exponent

    def __str__(self):
        return "Naive Bayes"

# K Nearest Neighbors classifier
class KNNClassifier(MLClassifier):
    def __init__(self, k=5):
        # Initialize the number of neighbors to consider
        self.k = k

    def fit(self, x, y):
        # Train the classifier by storing the training data
        self.x_train = x
        self.y_train = y

    def predict_single(self, input_features):
        # Predict the class of a single feature set
        distances = []
        for i, x in enumerate(self.x_train):
            # Calculate the Euclidean distance between the input features and each training instance
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(input_features, x)]))
            distances.append((distance, self.y_train[i]))
        # Sort the distances and get the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        # Count the class occurrences in the neighbors
        counts = {}
        for neighbor in neighbors:
            if neighbor[1] in counts:
                counts[neighbor[1]] += 1
            else:
                counts[neighbor[1]] = 1
        # Return the class with the highest count
        return max(counts, key=counts.get)

    def __str__(self):
        return f"K ({self.k}) Nearest Neighbors"

# Support Vector Machine classifier
class SVMClassifier(MLClassifier):
    def __init__(self, learning_rate=0.001, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])

        # Initialize weights and bias
        self.w = [0.0 for _ in range(n_features)]
        self.b = 0.0

        # Convert labels to +1 and -1 for binary classification
        y = [1 if label == "Good" else -1 for label in y]

        # Gradient descent
        for _ in range(n_samples):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (self.dot_product(x_i, self.w) - self.b) >= 1
                if condition:
                    for j in range(n_features):
                        self.w[j] -= self.learning_rate * (2 * self.lambda_param * self.w[j])
                else:
                    for j in range(n_features):
                        self.w[j] -= self.learning_rate * (2 * self.lambda_param * self.w[j] - x_i[j] * y[idx])
                    self.b -= self.learning_rate * y[idx]

    def predict_single(self, input_features):
        approx = self.dot_product(input_features, self.w) - self.b
        return "Good" if approx >= 0 else "Bad"

    def dot_product(self, x1, x2):
        return sum(x1[i] * x2[i] for i in range(len(x1)))

    def __str__(self):
        return "Support Vector Machine"
