import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, Normalizer


def normalize(x):
    # TODO: implement normalization. Should take column as input and return the normalized column
    norm = (x - x.min()) / (x.max() - x)  # general formula
    return norm


# TODO: update methods ot not use normalizer
# TODO: look into OneHotEncoder more and decide whether to make that method by hand

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

    # drop 'Id' (useless) and 'Class' (target)
    X = df.drop(['Id', 'Class'], axis=1)

    scaler = Normalizer()
    X = scaler.fit_transform(X)

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

    # drop id (useless) and type (the target)
    X = df.drop(['Id', 'Type'], axis=1)

    # the other classes are numeric, standardize
    scalar = Normalizer()
    X = scalar.fit_transform(X)

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

    y = df['Class'].map(class_mapping).values  # target is 'Class'

    X = df.drop('Class', axis=1)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)

    feature_names = encoder.get_feature_names_out(X.columns)
    X = pd.DataFrame(X_encoded, columns=feature_names)

    return X.values, y


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

    # new dataframe with encoded 'Sex' and numeric columns
    X = pd.concat([
        pd.DataFrame(sex_encoded, columns=sex_column_names),
        df[numeric_columns]
    ], axis=1)

    # standardize numerical values
    scaler = Normalizer()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

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

    numeric_columns = df.drop(['month', 'day', 'area'], axis=1).columns

    # new datasets with encoded month and day
    X = pd.concat([
        pd.DataFrame(categorical_encoded, columns=categorical_column_names),
        df[numeric_columns]
    ], axis=1)

    # standardize numerical values
    scaler = Normalizer()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

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

    # I still dropped names and erp because I don't think there are value in these columns
    # if you want me to keep them, we would need to encode the names. ERP is their model's prediction,
    # so I think it might cause overfitting problems
    numeric_columns = df.drop(['Vendor Name', 'Model Name', 'PRP', 'ERP'], axis=1).columns
    X = df[numeric_columns]

    # standardize continuous values
    scaler = Normalizer()
    X = pd.DataFrame(scaler.fit_transform(X), columns=numeric_columns)

    return X.values, y
