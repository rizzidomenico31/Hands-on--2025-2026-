import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('datasets/houses_augmented.csv')
print(df.describe())

df = df.drop('Id', axis=1)

# Looking for NaN values
nan_number = df.isna().sum()

# Drop NaN values

df.dropna(axis=0, inplace=True)
df = df.reset_index(drop=True)

# Is there any smarter solution to handle NaN?

# Looking for duplicates
df.drop_duplicates(inplace=True)

# Looking for categorical variables
categorical_columns = []
types = df.dtypes
categorical_columns = [df.columns[i] for i in range(len(types)) if types[i] == 'object']

values = {}

for c in categorical_columns:
    values[c] = df[c].unique().tolist()

for c in categorical_columns:
    for i, val in enumerate(values[c]):
        df[c].replace(val, i, inplace=True)

# Check if the conversion was succefully

_temp_cat = [df.columns[i] for i in range(len(df.dtypes)) if df.dtypes[i] == 'object']

correlated_columns = []

for col in df.corr().columns:
    if col == 'SalePrice':
        continue

    max_c = df.corr()[col].drop(col, axis=0).max()
        

    # if max_c > 0.3 and idx_correlated not in correlated_columns:
    if max_c > 0.3 and col not in correlated_columns:
        df.drop(col, axis=1, inplace=True)
        correlated_columns.append(idx_correlated)

df = df.reset_index(drop=True)
# Other possible useful pre-processing operations
df = df[df['YearBuilt'] > 1939].reset_index(drop=True)

features_names, label_name = df.columns[:-1], df.columns[-1]

x = df[features_names]
y = df[label_name]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

continuous_features = list(set(x.columns) - set(categorical_columns))

scaler = StandardScaler()
x_train[continuous_features] = scaler.fit_transform(x_train[continuous_features])
x_test[continuous_features] = scaler.transform(x_test[continuous_features])

x_train, x_test, y_train, y_test = x_train.values, x_test.values, y_train.values, y_test.values

linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)

y_pred = linear_regression.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(mse)
print(mae)
