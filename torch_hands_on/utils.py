import numpy as np
from collections import Counter

def outlier_hunt(df, target_col):
    outlier_indices = []

    Q1 = np.percentile(df[target_col], 25)
    Q3 = np.percentile(df[target_col], 75)
    IRQ = Q3 - Q1
    outlier_step = 1.5 * IRQ

    outlier_list_col = df[(df[target_col] < Q1 - outlier_step) |
                          (df[target_col] > Q3 + outlier_step)].index
    outlier_indices.extend(outlier_list_col)

    return outlier_indices