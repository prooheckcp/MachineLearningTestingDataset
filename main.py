# Imports
import math

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataFrame = pd.read_csv("17006903.csv")

# Constants
MODELS_DICTIONARY = {
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic regression model": LogisticRegression(max_iter=math.inf)
}

TEST_DATA_PERCENTAGE = 10  # %
AMOUNT_OF_ITERATIONS = 10

# Clear the dataFrame
X = dataFrame.drop(columns=['T'])  # known attributes
Y = dataFrame['T']  # goal data

# Getting test and train data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_DATA_PERCENTAGE / 100)

def convert_to_percentage(value):
    return str(round(value * 100, 2)) + "%"


def calculate_accuracy(model):
    # Training the model
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    # Testing accuracy
    return accuracy_score(Y_test, predictions)


# Tests a model and returns a tuple with the lowest score, highest and average
def testing_model(model):
    highest_score = -math.inf
    lowest_score = math.inf
    total_score = 0

    for index in range(AMOUNT_OF_ITERATIONS):
        score = calculate_accuracy(model)
        total_score += score

        if score > highest_score:
            highest_score = score

        if score < lowest_score:
            lowest_score = score

    average_score = total_score/AMOUNT_OF_ITERATIONS

    return lowest_score, highest_score, average_score


def print_model_results(lowest_score, highest_score, average_score):
    print("Lowest score: " + convert_to_percentage(lowest_score))
    print("Highest score: " + convert_to_percentage(highest_score))
    print("Average score: " + convert_to_percentage(average_score))
    print("\n")

for key, model in MODELS_DICTIONARY.items():
    print(key + ":")
    lowest_score, highest_score, average_score = testing_model(model)
    print_model_results(lowest_score, highest_score, average_score)