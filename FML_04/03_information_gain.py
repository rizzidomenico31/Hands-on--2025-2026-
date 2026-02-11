import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def calculate_entropy(labels):

    # Count how many are True (has diabetes, represented by 1)
    true_count = list(labels).count(1)

    # Count how many are False (no diabetes, represented by 0)
    false_count = list(labels).count(0)

    # Calculate the probability of each outcome
    prob_true = true_count / len(labels)
    prob_false = false_count / len(labels)

    # If a probability is 0 (e.g., all patients are 'True'), then check to avoid log(0)
    entropy = 0.0
    if prob_true > 0:
        entropy -= prob_true * np.log2(prob_true)
    if prob_false > 0:
        entropy -= prob_false * np.log2(prob_false)

    return entropy


def calculate_information_gain(feature_column, target_labels):

    starting_entropy = calculate_entropy(target_labels)

    # Since we are working with continuous data, we will simply split based on the mean value
    # In a real scenario with categorical data, we would split by each category
    split_threshold = feature_column.mean()

    # split the dataset into two groups based on the threshold because we are working with continuous data
    labels_below_threshold = target_labels[feature_column <= split_threshold]
    labels_above_threshold = target_labels[feature_column > split_threshold]

    # calculate the entropy for each group
    entropy_below = calculate_entropy(labels_below_threshold)
    entropy_above = calculate_entropy(labels_above_threshold)

    # calculate the weighted average of the new entropy after the split
    weight_below = len(labels_below_threshold) / len(target_labels)
    weight_above = len(labels_above_threshold) / len(target_labels)

    weighted_new_entropy = (weight_below * entropy_below) + (weight_above * entropy_above)

    # calculate the information gain for all features
    information_gain = starting_entropy - weighted_new_entropy

    return information_gain


# Load the diabetes dataset
diabetes = pd.read_csv('datasets/diabetes.csv')

# Get the list of all  features and target variable
all_feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']
target_column = diabetes['Outcome']

# store the score for each feature
feature_scores = {}

# Calculate the Information Gain for every single feature
for name in all_feature_names:
    feature_data = diabetes[name]
    score = calculate_information_gain(feature_data, target_column)
    feature_scores[name] = score

# sort the features by their scores in descending order
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)


print("Feature Ranking (Most Useful First) ")
for feature, score in sorted_features:
    print(f"Feature: {feature} | Score: {score:.4f}")


number_of_features_to_keep = 5
selected_features = [feature for feature, score in sorted_features[:number_of_features_to_keep]]
print(f"\n Selected features: {selected_features}\n")

############################################################################

# Select features and target variable
X = diabetes[selected_features].values
y = diabetes['Outcome'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Create a logistic regression model
logistic_model = LogisticRegression(random_state=42)
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'tol' : [0.1, 0.001]
}

grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy')

# Fit the model with grid search on the standardized training data
grid_search.fit(X_train_std, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Make predictions on the standardized test data using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_std)

# Evaluate the performance of the best model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print(f'Best Hyperparameters: {best_params}')
print(f'Accuracy with Best Model: {accuracy:.2f}')
print('Classification Report:\n', classification_report_str)
print('Confusion Matrix:\n', conf_matrix)
