import csv
import os
from random import shuffle, choice
import time

from normalizers import MinMaxNormalizer, ZScoreNormalizer, DecimalNormalizer
from classifiers import NaiveBayesClassifier, KNNClassifier, SVMClassifier

# Define a class to handle data operations
class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_csv(self):
        with open(self.filepath, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header row
            dataset = [row for row in csv_reader]
        return dataset

    def read_example(self):
        with open(self.filepath, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Read the header row
            example_line = choice(list(csv_reader))  # Choose a random line
        return header, example_line

    def train_test_split(self, dataset, test_size):
        shuffle(dataset)
        split_index = int(len(dataset) * (1 - test_size))
        return dataset[:split_index], dataset[split_index:]

    def separate_features_labels(self, dataset):
        features = [list(map(float, data[:-1])) for data in dataset]  # Exclude the label
        labels = [data[-1] for data in dataset]  # The label is the last element in each row
        return features, labels

    def get_file_timestamp(self, filepath):
        try:
            return os.path.getmtime(filepath)
        except OSError:
            return 0

    def write_normalized_data(self, normalized_data, filepath):
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(normalized_data)

    def read_normalized_data(self, filepath):
        with open(filepath, 'r') as file:
            csv_reader = csv.reader(file)
            normalized_data = [list(map(float, row)) for row in csv_reader]
        return normalized_data

    def preprocess_data(self, normalizer_classes):
        dataset = self.read_csv()
        train_data, test_data = self.train_test_split(dataset, test_size=0.2)
        train_features, train_labels = self.separate_features_labels(train_data)
        test_features, test_labels = self.separate_features_labels(test_data)

        normalized_train_features = {}
        normalized_test_features = {}

        for normalizer_class in normalizer_classes:
            normalizer_name = normalizer_class.__name__
            normalized_train_file = f'normalized_train_{normalizer_name}.csv'
            normalized_test_file = f'normalized_test_{normalizer_name}.csv'

            source_timestamp = self.get_file_timestamp(self.filepath)
            train_timestamp = self.get_file_timestamp(normalized_train_file)
            test_timestamp = self.get_file_timestamp(normalized_test_file)

            if train_timestamp > source_timestamp and test_timestamp > source_timestamp:
                # Read from cached normalized data
                normalized_train_features[normalizer_name] = self.read_normalized_data(normalized_train_file)
                normalized_test_features[normalizer_name] = self.read_normalized_data(normalized_test_file)
            else:
                # Normalize the data
                normalizer = normalizer_class()
                normalized_train = normalizer.normalize(train_features)
                normalized_test = normalizer.normalize(test_features)
                normalized_train_features[normalizer_name] = normalized_train
                normalized_test_features[normalizer_name] = normalized_test
                # Write normalized data to CSV files
                self.write_normalized_data(normalized_train, normalized_train_file)
                self.write_normalized_data(normalized_test, normalized_test_file)

        return normalized_train_features, train_labels, normalized_test_features, test_labels

# We want to store the results of each test run so we can compare them later
class TestResult:
    def __init__(self, normalizer_name, classifier_name, accuracy, precision, recall, f1_score, fit_time, predict_time):
        self._normalizer_name = normalizer_name
        self._classifier_name = classifier_name
        self._accuracy = accuracy
        self._precision = precision
        self._recall = recall
        self._f1_score = f1_score
        self._fit_time = fit_time
        self._predict_time = predict_time

    def __repr__(self):
        return (f"TestResult(normalizer_name={self._normalizer_name!r}, classifier_name={self._classifier_name!r}, "
                f"accuracy={self._accuracy:.2f}, precision={self._precision:.2f}, "
                f"recall={self._recall:.2f}, f1_score={self._f1_score:.2f}, "
                f"fit_time={self._fit_time:.2f}s, predict_time={self._predict_time:.2f}s)")

    # Getter methods
    @property
    def normalizer_name(self):
        return self._normalizer_name

    @property
    def classifier_name(self):
        return self._classifier_name

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def precision(self):
        return self._precision

    @property
    def recall(self):
        return self._recall

    @property
    def f1_score(self):
        return self._f1_score

    @property
    def fit_time(self):
        return self._fit_time

    @property
    def predict_time(self):
        return self._predict_time

# Define the banana object
class Banana:
    def __init__(self, size, weight, sweetness, softness, harvest_time, ripeness, acidity):
        self.size = size
        self.weight = weight
        self.sweetness = sweetness
        self.softness = softness
        self.harvest_time = harvest_time
        self.ripeness = ripeness
        self.acidity = acidity

    def get_features(self):
        return [self.size, self.weight, self.sweetness, self.softness, self.harvest_time, self.ripeness, self.acidity]

    def __str__(self):
        return (f"Banana(size={self.size}, weight={self.weight}, sweetness={self.sweetness}, softness={self.softness}, "
                f"harvest_time={self.harvest_time}, ripeness={self.ripeness}, acidity={self.acidity})")

    # Method to predict the quality of a banana
    def test_quality(self, classifier, normalizer=None):
        features = self.get_features()
        normalized_features = normalizer.normalize([features])[0]
        prediction = classifier.predict([normalized_features])[0]
        return prediction

    # Static method to encapsulate the banana input process
    @staticmethod
    def prompt_banana_details(header, example):
        print("Please enter the following details for the banana. Example values are shown in parentheses:")
        size = float(input(f"Enter {header[0]} ({example[0]}): "))
        weight = float(input(f"Enter {header[1]} ({example[1]}): "))
        sweetness = float(input(f"Enter {header[2]} ({example[2]}): "))
        softness = float(input(f"Enter {header[3]} ({example[3]}): "))
        harvest_time = float(input(f"Enter {header[4]} ({example[4]}): "))
        ripeness = float(input(f"Enter {header[5]} ({example[5]}): "))
        acidity = float(input(f"Enter {header[6]} ({example[6]}): "))
        return Banana(size, weight, sweetness, softness, harvest_time, ripeness, acidity)

# Define the main function to run the program
def main():

    # Define the path to the CSV file containing the Iris dataset
    filepath = '../banana_quality.csv'
    data_handler = DataHandler(filepath)
    
    # Get user input for normalizers
    normalizer_choice = input("Choose a normalizer: (1) Min-Max; (2) Z-Score; (3) Decimal; (4) All; (5) None: ")
    normalizer_classes = get_normalizer_classes(normalizer_choice)
    
    # Get user input for classifiers
    algorithm_choice = input("Choose an algorithm: (1) Naive Bayes; (2) KNN; (3) SVM; (4) All: ")
    classifiers = get_classifier_instances(algorithm_choice)

    # Check if KNNClassifier is in the list and replace add number of neighbors if necessary
    if any(isinstance(clf, KNNClassifier) for clf in classifiers):
        k = int(input("Enter the number of neighbors for KNN: "))
        classifiers = [KNNClassifier(k) if isinstance(clf, KNNClassifier) else clf for clf in classifiers]

    # Preprocess data with the chosen normalizers
    normalized_train_features, train_labels, normalized_test_features, test_labels = data_handler.preprocess_data(normalizer_classes)

    test_results = []
    trained_models = {}

    # Iterate through each normalized dataset
    for normalizer_name, train_features in normalized_train_features.items():
        for classifier in classifiers:
            # Print the training message without a newline
            print(f"Training {classifier} with {normalizer_name}...", end='')
            
            # Initialize and train the classifier
            start_time = time.time()
            classifier.fit(train_features, train_labels)
            fit_time = time.time() - start_time
            
            # Save the trained classifier for later use
            trained_models[(normalizer_name, str(classifier))] = classifier
            
            # Predict on the corresponding test set
            start_time = time.time()
            test_features = normalized_test_features[normalizer_name]
            predictions = classifier.predict(test_features)
            predict_time = time.time() - start_time
            
            # Calculate total time
            total_time = fit_time + predict_time
            
            # Generate the classification report
            report = classifier.classification_report(test_labels, predictions)
            
            # Calculate overall metrics using the new method
            accuracy, precision, recall, f1_score = classifier.overall_metrics(report)
            
            # Create a TestResult object and store it
            result = TestResult(normalizer_name, str(classifier), accuracy, precision, recall, f1_score, fit_time, predict_time)
            test_results.append(result)
            
            # Print the elapsed time
            print(f" Done. Total time: {total_time:.2f}s")

    # Sort the results by accuracy in descending order
    test_results.sort(key=lambda x: x.accuracy, reverse=True)

    # Print header for results
    print("\n{:<25} {:<25} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Classifier", "Normalizer", "Accuracy", "Precision", "Recall", "F1 Score", "Train(s)", "Predict(s)"
    ))

    # Display the sorted results
    for result in test_results:
        print("{:<25} {:<25} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<15.2f} {:<15.2f}".format(
            result.classifier_name, result.normalizer_name, result.accuracy, result.precision, result.recall, result.f1_score, result.fit_time, result.predict_time
        ))

    while input("Would you like to test a banana? (y/n): ").lower() == 'y':
        # Read the header and an example line from the CSV file
        header, example_line = data_handler.read_example()

        # Predict the quality of a new banana
        new_banana = Banana.prompt_banana_details(header, example_line)
        banana_features = new_banana.get_features()

        for (normalizer_name, classifier_name), classifier in trained_models.items():
            normalizer = next(n for n in normalizer_classes if n.__name__ == normalizer_name)()
            normalized_banana_features = normalizer.normalize([banana_features])[0]
            prediction = classifier.predict([normalized_banana_features])[0]
            print(f"Using {normalizer_name} and {classifier_name} classifier, the banana is predicted to be: {prediction}")

    print()

def get_normalizer_classes(choice):
    match choice:
        case "1":
            return [MinMaxNormalizer]
        case "2":
            return [ZScoreNormalizer]
        case "3":
            return [DecimalNormalizer]
        case "4":
            return [MinMaxNormalizer, ZScoreNormalizer, DecimalNormalizer]
        case "5":
            return []

def get_classifier_instances(choice):
    match choice:
        case "1":
            return [NaiveBayesClassifier()]
        case "2":
            return [KNNClassifier()]
        case "3":
            return [SVMClassifier()]
        case "4":
            return [NaiveBayesClassifier(), KNNClassifier(), SVMClassifier()]

def display_results(classifier, report, fit_time, predict_time):
    print(f"\n{classifier}:\n")
    for label, metrics in report.items():
        print(f"\tClass {label}:")
        for metric, value in metrics.items():
            print(f"\t\t{metric}: {value:.2f}")
        print()
    print(f"\tFit time: {fit_time:.4f}s")
    print(f"\tPrediction time: {predict_time:.4f}s")

def get_test_ratio():
    while True:
        try:
            test_percentage = float(input("What ratio of data would you like to use for testing (0.01-0.99)?"))
            if 0.01 <= test_percentage <= 0.99:
                return test_percentage
            else:
                print("Please enter a valid percentage between 0.01 and 0.99.")
        except ValueError:
            print("Invalid input. Please enter a number between 0.01 and 0.99.")

# This block checks if this script is the main program and runs the main function
if __name__ == "__main__":
    main()