# normalizers.py

import math

# Define a base class for normalizers so we can reuse the normalize method
class Normalizer:
    def __init__(self):
        pass

    def normalize(self, features):
        raise NotImplementedError("This method should be overridden by subclasses.")

# The min-max normalizer scales features to a specified range
class MinMaxNormalizer(Normalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, features):
        columns = list(zip(*features))
        min_vals = [min(column) for column in columns]
        max_vals = [max(column) for column in columns]
        normalized_features = []

        for row in features:
            normalized_row = [
                (row[i] - min_vals[i]) / (max_vals[i] - min_vals[i]) if (max_vals[i] - min_vals[i]) != 0 else 0
                for i in range(len(row))
            ]
            normalized_features.append(normalized_row)

        return normalized_features

# The Z-score normalizer scales features based on the mean and standard deviation
class ZScoreNormalizer(Normalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, features):
        columns = list(zip(*features))
        means = [sum(column) / len(column) for column in columns]
        std_devs = [((sum([(x - mean) ** 2 for x in column]) / len(column)) ** 0.5) for column, mean in zip(columns, means)]
        normalized_features = []

        for row in features:
            normalized_row = [
                (row[i] - means[i]) / std_devs[i] if std_devs[i] != 0 else 0
                for i in range(len(row))
            ]
            normalized_features.append(normalized_row)

        return normalized_features

# The decimal normalizer scales features based on the maximum absolute value
class DecimalNormalizer(Normalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, features):
        columns = list(zip(*features))
        max_abs_vals = [max(abs(x) for x in column) for column in columns]
        normalized_features = []

        for row in features:
            normalized_row = [
                row[i] / max_abs_vals[i] if max_abs_vals[i] != 0 else 0
                for i in range(len(row))
            ]
            normalized_features.append(normalized_row)

        return normalized_features
