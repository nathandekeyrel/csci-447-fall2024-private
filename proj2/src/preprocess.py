import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif

"""
NOTE: this does rely on some sklearn methods. I will talk to either the prof or TA (whichever is in class) to make sure
it is ok to use them in this sense. If not, the majority of the datasets won't need feature selection (assumption)
so I will figure out how to do it from scratch for the forest-fires dataset if needed. Because of how explicit the names
file was about feature correlation, I think it's worth checking for that dataset.
"""


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
    y = df['Class'].map(class_mapping).values

    # drop 'Id' (useless) and 'Class' (target)
    X = df.drop(['Id', 'Class'], axis=1)

    scaler = StandardScaler()
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

    # drop id (useless) and type (the target)
    X = df.drop(['Id', 'Type'], axis=1)
    # Target is 'Type' is valued from 1-7 (integers)
    # 4 vehicle_windows_non_float_processed (none in this database)
    y = df['Type'].values

    # the other classes are numeric, standardize
    scalar = StandardScaler()
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

    # these columns either have values of all 0 or all 1. depending on outcome we may or may not keep this.
    columns_to_drop = [
        'Date', 'Temp', 'Leafspots Halo', 'Leaf Shread', 'Leaf Malf', 'Leaf Mild',
        'Stem', 'Seed', 'Mold Growth', 'Seed Discolor', 'Seed Size', 'Shriveling'
    ]

    class_mapping = {
        'D1': 0,
        'D2': 1,
        'D3': 2,
        'D4': 3
    }

    y = df['Class'].map(class_mapping).values  # target is 'Class'

    # temporary to check feature importance
    X = df.drop('Class', axis=1)

    # check feature importance
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    print("\nFeature importance:")
    print(mi_scores.sort_values(ascending=False))

    # final X will be determined by mi_scores output.

    # X = df.drop([columns_to_drop, 'Class'], axis=1)
    encoder = OneHotEncoder(handle_unknown='ignore')
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

    # Target: 'Rings' --> Age = Rings + 1.5
    y = df['Rings'].values

    # one-hot encode only the 'Sex' column
    encoder = OneHotEncoder(handle_unknown='ignore')
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

    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    # check feature importance
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    print("\nFeature importance:")
    print(mi_scores.sort_values(ascending=False))

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
    y = np.log1p(df['area'])
    # I used log1p because "This function is particularly useful when xxx is close to zero"
    # ref - https://medium.com/@noorfatimaafzalbutt/understanding-np-log-and-np-log1p-in-numpy-99cefa89cd30
    # documentation - https://numpy.org/doc/2.0/reference/generated/numpy.log1p.html

    # one-hot encode 'month' and 'day'
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_encoded = encoder.fit_transform(df[['month', 'day']])
    categorical_column_names = encoder.get_feature_names_out(['month', 'day'])

    numeric_columns = df.drop(['month', 'day', 'area'], axis=1).columns

    X = pd.concat([
        pd.DataFrame(categorical_encoded, columns=categorical_column_names),
        df[numeric_columns]
    ], axis=1)

    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    # Note: several of the attributes may be correlated, thus it makes sense to apply some sort of feature selection.
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    print("\nFeature importance:")
    print(mi_scores.sort_values(ascending=False))

    return X.values, y.values


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

    # drop vendor name, model name, and erp. they don't actually tell us anything about the computer
    # also dropped PRP as it's the target.
    numeric_columns = df.drop(['Vendor Name', 'Model Name', 'PRP', 'ERP'], axis=1).columns
    X = df[numeric_columns]

    # standardize continuous values
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=numeric_columns)

    # at least check if correlation. Unlikely we need it here.
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    print("\nFeature importance:")
    print(mi_scores.sort_values(ascending=False))

    return X.values, y
