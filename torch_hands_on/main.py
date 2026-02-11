import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from nn import NN
from sklearn.model_selection import train_test_split
import torch
from utils import outlier_hunt
import math
from itertools import product
np.random.seed(42)
torch.manual_seed(42)

def cross_val_generator(k, x_training, y_training):
    for i in range(k):
        start = int(math.floor(i * x_training.shape[0] / k))
        stop = int(math.floor((i + 1) * x_training.shape[0] / k))

        x_val_i, y_val_i = x_training[start:stop], y_training[start:stop]
        x_train_i, y_train_i = np.delete(x_training, range(start, stop), axis=0), \
                               np.delete(y_training, range(start, stop))

        yield x_train_i, y_train_i, x_val_i, y_val_i

df = pd.read_csv('data/ds_salaries.csv')
print(df.columns)
df = df.sample(frac=1).reset_index(drop=True)
df = df.dropna().reset_index(drop=True)
df = df.drop_duplicates().reset_index(drop=True)

target = 'salary'
X = df.drop(target, axis=1)
y = df['salary']

columns_to_encode = ['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'remote_ratio', 'company_size', 'salary_currency']

continuous_features = X.select_dtypes(include=["float64", "int64"]).columns
categorical_features = X.select_dtypes(exclude=["float64", "int64"]).columns

continuous_features = ['salary_in_usd']
categorical_features = categorical_features.tolist() + ['work_year', 'remote_ratio']

for col in categorical_features:
    label_encoder = LabelEncoder()
    X[col] = label_encoder.fit_transform(X[col])

outlier_indices = outlier_hunt(X, 'salary_in_usd')
X = X.drop(outlier_indices).reset_index(drop=True)
y = y.drop(outlier_indices).reset_index(drop=True)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for col in continuous_features:
    st = StandardScaler()
    x_train[col] = st.fit_transform(x_train[[col]])
    x_test[col] = st.transform(x_test[[col]])

x_train, y_train = torch.from_numpy(x_train.values).float(), torch.from_numpy(y_train.values).float()
x_test, y_test = torch.from_numpy(x_test.values).float(), torch.from_numpy(y_test.values).float()
criterion = torch.nn.MSELoss()

epochs = 100
lr_values = [0.001, 0.01]
l2_values = [0.1, 0.01]
k = 5
params = list(product(lr_values, l2_values))
best_mse, best_hp = np.inf, None

for lr, l2 in params:
    running_mse, fold_mse = 0, 0
    for i, sets in enumerate(cross_val_generator(k, x_train, y_train)):
        xt, yt, xv, yv = sets
        model = NN(input_size=x_train.shape[1], hidden_size=[5, 3], output_size=1, epochs=epochs, lr=lr, l2=l2, criterion=criterion)
        val_metric = model.fit(xt, yt, xv, yv)
        running_mse += val_metric
    fold_mse = running_mse / k

    print(f"current situation: {lr}, {l2}, {fold_mse}")

    if fold_mse < best_mse:
        best_mse = fold_mse
        best_hp = (lr, l2)

print(f"Best val mse: {best_mse}, best hp: {best_hp}")

model = NN(input_size=x_train.shape[1], hidden_size=[5, 3], output_size=1, epochs=epochs, lr=best_hp[0], l2=best_hp[1], criterion=criterion)
test_mse = model.fit(x_train, y_train, x_test, y_test)

print(test_mse)

