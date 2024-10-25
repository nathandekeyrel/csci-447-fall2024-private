import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(filepath):
    """Preprocess a dataset based on its filename.

    :param filepath: Path to the dataset file
    :return: Preprocessed dataset
    """
    if 'breast-cancer-wisconsin' in filepath:
        return _preprocess_cancer(filepath)
    elif 'glass' in filepath:
        return _preprocess_glass(filepath)
    elif 'soybean-small' in filepath:
        return _preprocess_soybean(filepath)
    elif 'abalone' in filepath:
        return _preprocess_abalone(filepath)
    elif 'forestfires' in filepath:
        return _preprocess_fires(filepath)
    elif 'machine' in filepath:
        return _preprocess_computer(filepath)
    else:
        print(f"Bad dataset: {filepath}")


def normalize(x):
    """Normalize array values to range [0,1] using min-max scaling.

    :param x: Array-like input data to normalize
    :return: Normalized array with values scaled to [0,1]
    """
    norm = (x - x.min()) / (x.max() - x.min())
    return norm


def normalize_numeric_columns(df, columns):
    """Normalize specified numeric columns in a dataframe to range [0,1].

    :param df: Input pandas DataFrame
    :param columns: List of column names to normalize
    :return: DataFrame with specified columns normalized
    """
    df_normalized = df.copy()
    for col in columns:
        df_normalized[col] = normalize(df[col])
    return df_normalized


def _preprocess_cancer(filepath):
    """Preprocess Breast Cancer dataset.

    :param filepath: Path to the data file
    :return: Tuple of (X, y) where X is the feature matrix and y is the target vector
    """
    df = pd.read_csv(filepath, header=None, na_values='?')
    # give columns a name (I'm slow, so it was better than trying to index into the right one)
    columns = ['Id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
               'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
               'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    df.columns = columns

    df = df.dropna()  # there are 16 missing values, drop them because instance count is high

    # map class values to binary
    class_mapping = {
        2: 0,
        4: 1
    }

    y = df['Class'].map(class_mapping).values  # Target is 'Class'

    numeric_columns = [col for col in df.columns if col not in ['Id', 'Class']]
    df_normalized = normalize_numeric_columns(df, numeric_columns)
    X = df_normalized[numeric_columns].values

    return X, y


def _preprocess_glass(filepath):
    """Preprocess Glass dataset.

    :param filepath: Path to the data file.
    :return: Tuple of (X, y) where X is the feature matrix and y is the target vector
    """
    df = pd.read_csv(filepath, header=None, na_values='?')

    columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    df.columns = columns

    # target is 'Type' is valued from 1-7 (integers)
    # "4 vehicle_windows_non_float_processed (none in this database)" - from names

    class_mapping = {
        1: 0,
        2: 1,
        3: 2,
        5: 3,
        6: 4,
        7: 5
    }
    y = df['Type'].map(class_mapping).values

    numeric_columns = [col for col in df.columns if col not in ['Id', 'Type']]
    df_normalized = normalize_numeric_columns(df, numeric_columns)
    X = df_normalized[numeric_columns].values

    return X, y


def _preprocess_soybean(filepath):
    """Preprocess Soybean dataset.

    :param filepath: Path to the data file.
    :return: Tuple of (X, y) where X is the feature matrix and y is the target vector
    """
    df = pd.read_csv(filepath, header=None)

    columns = ['Date', 'Plant Stand', 'Precip', 'Temp', 'Hail', 'Crop-hist', 'Area Damaged', 'Severity',
               'Seed Tmt', 'Germination', 'Plant Growth', 'Leaves', 'Leafspots Halo', 'Leafspots Marg',
               'Leafspot Size', 'Leaf Shread', 'Leaf Malf', 'Leaf Mild', 'Stem', 'Lodging',
               'Stem Cankers', 'Canker Lesion', 'Fruiting Bodies', 'External Decay', 'Mycelium', 'Interior Discolor',
               'Sclerotia', 'Fruit Pods', 'Fruit Spots', 'Seed', 'Mold Growth', 'Seed Discolor', 'Seed Size',
               'Shriveling', 'Roots', 'Class']
    df.columns = columns

    class_mapping = {
        'D1': 0,
        'D2': 1,
        'D3': 2,
        'D4': 3
    }
    y = df['Class'].map(class_mapping).values  # target is class

    X = df.drop('Class', axis=1)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X = encoder.fit_transform(X)

    return X, y


def _preprocess_abalone(filepath):
    """Preprocess Abalone dataset...

    :param filepath: Path to the data file.
    :return: Tuple of (X, y) where X is the feature matrix and y is the target vector
    """
    df = pd.read_csv(filepath, header=None)

    columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight',
               'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
    df.columns = columns

    # target: 'Rings'
    y = df['Rings'].values

    # one-hot encode only the 'Sex' column
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sex_encoded = encoder.fit_transform(df[['Sex']])
    sex_column_names = encoder.get_feature_names_out(['Sex'])

    # define numeric columns
    numeric_columns = ['Length', 'Diameter', 'Height', 'Whole Weight',
                       'Shucked Weight', 'Viscera Weight', 'Shell Weight']
    df_normalized = normalize_numeric_columns(df, numeric_columns)

    # new dataframe with encoded 'Sex' and normalized numeric columns
    X = pd.concat([
        pd.DataFrame(sex_encoded, columns=sex_column_names),
        df_normalized[numeric_columns]
    ], axis=1)

    return X.values, y


def _preprocess_fires(filepath):
    """Preprocess Forest Fires dataset...

    :param filepath: Path to the data file.
    :return: Tuple of (X, y) where X is the feature matrix and y is the target vector
    """
    df = pd.read_csv(filepath)

    columns = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC',
               'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
    df.columns = columns

    # Target: area - the burned area of the forest (in ha): 0.00 to 1090.84
    #                (this output variable is very skewed towards 0.0, thus it may make
    #                sense to model with the logarithm transform)
    y = np.log1p(df['area']).values
    # I used log1p because "This function is particularly useful when xxx is close to zero"
    # ref - https://medium.com/@noorfatimaafzalbutt/understanding-np-log-and-np-log1p-in-numpy-99cefa89cd30
    # documentation - https://numpy.org/doc/2.0/reference/generated/numpy.log1p.html

    # one-hot encode 'month' and 'day'
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_encoded = encoder.fit_transform(df[['month', 'day']])
    categorical_column_names = encoder.get_feature_names_out(['month', 'day'])

    numeric_columns = [col for col in df.columns if col not in ['month', 'day', 'area']]
    df_normalized = normalize_numeric_columns(df, numeric_columns)

    # new datasets with normalized values and encoded month and day
    X = pd.concat([
        pd.DataFrame(categorical_encoded, columns=categorical_column_names),
        df_normalized[numeric_columns]
    ], axis=1)

    return X.values, y


def _preprocess_computer(filepath):
    """Preprocess Computer dataset...

    :param filepath: Path to the data file.
    :return: Tuple of (X, y) where X is the feature matrix and y is the target vector
    """
    df = pd.read_csv(filepath, header=None)

    columns = ['Vendor Name', 'Model Name', 'MYCT', 'MMIN',
               'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
    df.columns = columns

    y = df['PRP'].values  # Target: 'PRP', continuously valued

    # encode vendor names
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    vendor_encoded = encoder.fit_transform(df[['Vendor Name']])
    vendor_columns = encoder.get_feature_names_out(['Vendor Name'])

    # I still dropped model name and erp because I don't think there are value in these columns

    numeric_columns = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']
    df_normalized = normalize_numeric_columns(df, numeric_columns)

    # new set with encoded vendors with normalized numeric features
    X = pd.concat([
        pd.DataFrame(vendor_encoded, columns=vendor_columns),
        df_normalized[numeric_columns]
    ], axis=1)

    return X.values, y
