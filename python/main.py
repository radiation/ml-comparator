import csv
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

    def preprocess_data(self, normalizer_classes):
        dataset = self.read_csv()
        train_data, test_data = self.train_test_split(dataset, test_size=0.2)
        train_features, train_labels = self.separate_features_labels(train_data)
        test_features, test_labels = self.separate_features_labels(test_data)

        normalized_train_features = {normalizer_class.__name__: normalizer_class().normalize(train_features) for normalizer_class in normalizer_classes}
        normalized_test_features = {normalizer_class.__name__: normalizer_class().normalize(test_features) for normalizer_class in normalizer_classes}

        return normalized_train_features, train_labels, normalized_test_features, test_labels

# We want to store the results of each test run so we can compare them later
class TestResult:
    def __init__(self, normalizer_name, classifier_name, accuracy, precision, recall, f1_score):
        self.__normalizer_name = normalizer_name
        self.__classifier_name = classifier_name
        self.__accuracy = accuracy
        self.__precision = precision
        self.__recall = recall
        self.__f1_score = f1_score

    def __str__(self):
        return (f"Normalizer: {self.__normalizer_name}, Classifier: {self.__classifier_name}, "
                f"Accuracy: {self.__accuracy:.2f}, Precision: {self.__precision:.2f}, "
                f"Recall: {self.__recall:.2f}, F1 Score: {self.__f1_score:.2f}")

    # Getter methods
    @property
    def normalizer_name(self):
        return self.__normalizer_name

    @property
    def classifier_name(self):
        return self.__classifier_name

    @property
    def accuracy(self):
        return self.__accuracy

    @property
    def precision(self):
        return self.__precision

    @property
    def recall(self):
        return self.__recall

    @property
    def f1_score(self):
        return self.__f1_score

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
    normalizer_choice = input("Choose a normalizer: (1) Min-Max (2) Z-Score (3) Decimal (4) All (5) None: ")
    normalizer_classes = get_normalizer_classes(normalizer_choice)
    
    # Get user input for classifiers
    algorithm_choice = input("Choose an algorithm: (1) Naive Bayes (2) KNN (3) SVM (4) All: ")
    classifiers = get_classifier_instances(algorithm_choice)
    
    # Preprocess data with the chosen normalizers
    normalized_train_features, train_labels, normalized_test_features, test_labels = data_handler.preprocess_data(normalizer_classes)

    test_results = []

    # Iterate through each normalized dataset
    for normalizer_name, train_features in normalized_train_features.items():
        print(f"Training with {normalizer_name} normalization:")
        
        # Iterate through each classifier
        for classifier in classifiers:

            # Add result to test_results

            start_time = time.time()
            # Initialize and train the classifier
            classifier.fit(train_features, train_labels)
            fit_time = time.time() - start_time

            # Predict on the corresponding test set
            test_features = normalized_test_features[normalizer_name]
            start_time = time.time()
            predictions = classifier.predict(test_features)
            predict_time = time.time() - start_time

            # Generate and print the classification report
            report = classifier.classification_report(test_labels, predictions)

            for label, metrics in report.items():
                print(f"Class {label}:")
                for metric, value in metrics.items():
                    print(f"\t{metric}: {value:.2f}")
            print()

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