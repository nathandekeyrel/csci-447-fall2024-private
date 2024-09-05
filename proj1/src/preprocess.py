import pandas as pd
import numpy as np
import math
from scipy import stats


def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    if 'house-votes-84' not in filepath:
        df = df.replace('?', np.nan)
    df = remove_missing_values(df)
    df = discretize_continuous_values(df)

    df_with_noise = create_noise(df.copy())

    return df, df_with_noise


def remove_missing_values(df):
    df = df.dropna(how='any')
    return df


def get_num_bins(data):
    num_bins = max(2, int(math.ceil(math.log2(len(data)) + 1)))
    return num_bins


def discretize_continuous_values(df):
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            n_bins = get_num_bins(df[column])
            unique_values = df[column].unique()

            if len(unique_values) <= n_bins:
                bins = sorted(unique_values)
            else:
                mean = df[column].mean()
                std = df[column].std()

                bins_edges = stats.norm.ppf(np.linspace(0, 1, n_bins + 1), loc=mean, scale=std)
                bins_edges = np.clip(bins_edges, df[column].min(), df[column].max())

                bins = sorted(set(bins_edges))

            if len(bins) > 1:
                df[column] = pd.qcut(df[column], q=len(bins) - 1, labels=False, duplicates='drop')
            else:
                df[column] = 0

    return df


def create_noise(df):
    num_of_features = len(df.columns)
    number_of_noise_features = max(1, int(0.1 * num_of_features))

    noise_features = np.random.choice(df.columns, number_of_noise_features, replace=False)

    for feature in noise_features:
        df[feature] = np.random.permutation(df[feature].values)

    return df
