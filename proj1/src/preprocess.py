import pandas as pd
import numpy as np
import math
from scipy import stats


def preprocess_data(filepath):
    """
    Preprocess the dataset from a given filepath in main

    This function performs the following:
    1. Read the csv fie
    2. Replaces '?' with NaN in every dataset except 'house-votes-84'
        NOTE: house-votes-84 '?' value does not represent missing values
    3. Remove missing values
    4. Discretize continuous variables
    5. Create noisy version of the dataset

    :param filepath: Path to the data file
    :return: Tuple containing two pandas dataframes:
        - The first: The preprocessed original dataset
        - The second: The preprocessed dataset with added noise
    """
    df = pd.read_csv(filepath, header=None)
    if 'house-votes-84' not in filepath:
        df = df.replace('?', np.nan)
    df = remove_missing_values(df)
    df = discretize_continuous_values(df)

    df_with_noise = create_noise(df.copy())

    return df, df_with_noise


def remove_missing_values(df):
    """
    Remove missing values from a dataframe

    :param df: Pandas dataframe derived from dataset
    :return: Pandas dataframe without missing values
    """
    df = df.dropna(how='any')
    return df


def discretize_continuous_values(df):
    """
    Processes each column of the dataframe. For continuous columns
    (int64, float64), split into quartiles.

    :param df: Pandas dataframe with no missing values
    :return: Pandas dataframe with properly discretized continuous values
    """
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df[column] = pd.qcut(df[column], q=4, labels=False, duplicates='drop')
    return df


def create_noise(df):
    """
    This function selects no more than 10% of the features randomly
    and shuffles their values, introducing noise into the dataset.

    :param df: Pandas dataframe post-processing
    :return:  Pandas dataframe with added noise in 10% of features
    """
    num_of_features = len(df.columns)
    number_of_noise_features = max(1, math.floor(.1 * num_of_features))

    noise_features = np.random.choice(df.columns, number_of_noise_features, replace=False)

    for feature in noise_features:
        df[feature] = np.random.permutation(df[feature].values)

    return df
