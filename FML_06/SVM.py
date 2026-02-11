from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Load dataset
data = datasets.load_iris(as_frame=True)
X = data.data
y = data.target

# Separate continuous and categorical features
continuous_features = X.select_dtypes(include=["float64", "int64"]).columns
categorical_features = X.select_dtypes(exclude=["float64", "int64"]).columns

# Split the data into train and test sets
X_train_continuos_scaled, X_test_continuos_scaled, y_train, y_test = train_test_split(pd.DataFrame(X), y, test_size=0.2, random_state=42)

# Scale only continuous features
scaler = StandardScaler()

X_train_continuos_scaled[continuous_features] = scaler.fit_transform(X_train_continuos_scaled[continuous_features])
X_test_continuos_scaled[continuous_features] = scaler.transform(X_test_continuos_scaled[continuous_features])

# Train an SVM
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train_continuos_scaled, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test_continuos_scaled)
print(classification_report(y_test, y_pred))
