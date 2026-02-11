import numpy as np  # linear algebra
import pandas as pd  # read and wrangle dataframes
import seaborn as sns  # statistical visualizations and aesthetics
from sklearn.preprocessing import (FunctionTransformer, StandardScaler)  # preprocessing
from sklearn.decomposition import PCA  # dimensionality reduction
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV,
                                     learning_curve)  # model selection modules
import warnings
from sklearn.svm import SVC
from utils import plot_skew, plot_learning_curve, outlier_hunt
import math
from itertools import product
from sklearn.metrics import accuracy_score


def cross_val_generator(k, x_training, y_training):
    for i in range(k):
        start = int(math.floor(i * x_training.shape[0] / k))
        stop = int(math.floor((i + 1) * x_training.shape[0] / k))

        x_val_i, y_val_i = x_training[start:stop], y_training[start:stop]
        x_train_i, y_train_i = np.delete(x_training, range(start, stop), axis=0), \
                               np.delete(y_training, range(start, stop))

        yield x_train_i, y_train_i, x_val_i, y_val_i


warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

df = pd.read_csv('data/glass.csv')
features = df.columns[:-1].tolist()
print(df.shape)
print(df.head(5))

outlier_indices = outlier_hunt(df[features])
df = df.drop(outlier_indices).reset_index(drop=True)
print(df.shape)

X = df[features]
y = df['Type']

test_size = 0.2
seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# We identify how many components are needed to contain a variance above a threshold of about 80%.
# For this purpose we use a comulative sum to identify the optimal value.
pca = PCA(n_components=X_train.shape[1], random_state=seed)
pca.fit(X_train)
var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

# Cumulative variance explained
for i, sum in enumerate(cum_var_exp):
    print("PC" + str(i+1), f"Cumulative variance: {cum_var_exp[i]*100} %")

# Pay attention to the order! It is important to perform normalization first and then
# feature selection based on the components identified with PCA.

pca = PCA(n_components=5, random_state=seed)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

C_list = [10, 50, 100]
kernel_list = ['linear', 'rbf', 'sigmoid']
tol_list = [1e-2, 1e-3, 1e-4]

param_grid = list(product(C_list, kernel_list, tol_list))

k = 10

best_accuracy, best_hp = 0, None

for c, kernel, tol in param_grid:
    running_accuracy, fold_accuracy = 0, 0
    for i, sets in enumerate(cross_val_generator(k, X_train, y_train)):
        xt, yt, xv, yv = sets
        svm = SVC(C=c, kernel=kernel, tol=tol)
        svm.fit(xt, yt)
        yv_pred = svm.predict(xv)
        acc = accuracy_score(yv, yv_pred)
        running_accuracy += acc
    fold_accuracy = running_accuracy / k

    if fold_accuracy > best_accuracy:
        best_accuracy = fold_accuracy
        best_hp = (c, kernel, tol)


print(f"Best params: {best_hp}, best accuracy: {best_accuracy}")

svm = SVC(C=best_hp[0], kernel=best_hp[1], tol=best_hp[2])
svm.fit(X_train, y_train)
preds = svm.predict(X_test)
test_acc = accuracy_score(y_test, preds)

print(f"Final test acc: {test_acc}")
