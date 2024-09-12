import pandas as pd
import numpy as np


def preprocess_data(filepath):
    """
    Preprocess a dataset based on its filename.

    :param filepath: Path to the dataset file
    :return: Preprocessed dataset
    """
    if 'breast-cancer-wisconsin' in filepath:
        return preprocess_cancer(filepath)
    elif 'glass' in filepath:
        return preprocess_glass(filepath)
    elif 'house-votes-84' in filepath:
        return preprocess_votes(filepath)
    elif 'iris' in filepath:
        return preprocess_iris(filepath)
    elif 'soybean-small' in filepath:
        return preprocess_soybean(filepath)
    else:
        print(f"Bad dataset: {filepath}")


def preprocess_cancer(filepath):
    """
    Preprocess the breast cancer dataset.

    :param filepath: Path to the cancer dataset file
    :return: Features (X), target (y) as numpy arrays, number of unique classes in target
    """
    df = pd.read_csv(filepath, header=None, na_values='?')

    columns = ['Id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
               'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
               'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    df.columns = columns

    df = df.dropna()

    df = df.drop('Id', axis=1)

    for col in df.columns[:-1]:
        df[col] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')

    class_mapping = {
        2: 0,
        4: 1
    }

    df['Class'] = df['Class'].map(class_mapping)

    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    num_classes = len(np.unique(y))

    return X, y, num_classes


def preprocess_glass(filepath):
    """
    Preprocess the glass dataset.

    :param filepath: Path to the glass dataset file
    :return: Features (X), target (y) as numpy arrays, number of unique classes in target
    """
    df = pd.read_csv(filepath, header=None, na_values='?')

    columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    df.columns = columns

    df = df.drop('Id', axis=1)

    X = df.drop('Type', axis=1)
    y = df['Type']

    for column in X.columns:
        X[column] = pd.qcut(X[column], q=5, labels=False, duplicates='drop')

    X = X.values
    y = y.values

    num_classes = len(np.unique(y))

    return X, y, num_classes


def preprocess_votes(filepath):
    """
    Preprocess the votes dataset.

    :param filepath: Path to the votes dataset file
    :return: Features (X), target (y) as numpy arrays, number of unique classes in target
    """
    df = pd.read_csv(filepath, header=None)

    columns = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing',
               'adoption-of-the-budget-resolution', 'physician-fee-freeze',
               'el-salvador-aid', 'religious-groups-in-schools',
               'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
               'mx-missile', 'immigration', 'synfuels-corporation-cutback',
               'education-spending', 'superfund-right-to-sue', 'crime',
               'duty-free-exports', 'export-administration-act-south-africa']
    df.columns = columns

    df = df.replace({'y': 1, 'n': 0, '?': 2})

    class_mapping = {
        'democrat': 0,
        'republican': 1
    }

    df['Class Name'] = df['Class Name'].map(class_mapping)

    X = df.drop('Class Name', axis=1).values
    y = df['Class Name'].values

    num_classes = len(np.unique(y))

    return X, y, num_classes


def preprocess_iris(filepath):
    """
    Preprocess the iris dataset.

    :param filepath: Path to the iris dataset file
    :return: Features (X), target (y) as numpy arrays, number of unique classes in target
    """
    df = pd.read_csv(filepath, header=None)

    columns = ['Sepal Width', 'Sepal Length', 'Petal Length', 'Petal Width', 'Class']
    df.columns = columns

    for column in df.columns[:-1]:
        df[column] = pd.qcut(df[column], q=5, labels=False, duplicates='drop')

    class_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

    df['Class'] = df['Class'].map(class_mapping)

    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    num_classes = len(np.unique(y))

    return X, y, num_classes


def preprocess_soybean(filepath):
    """
    Preprocess the soybean dataset. Drop columns with only 0 values

    :param filepath: Path to the soybean dataset file
    :return: Features (X), target (y) as numpy arrays, number of unique classes in target
    """
    df = pd.read_csv(filepath, header=None)

    columns = ['Date', 'Plant Stand', 'Precip', 'Temp', 'Hail', 'Crop-hist', 'Area Damaged', 'Severity',
               'Seed Tmt', 'Germination', 'Plant Growth', 'Leaves', 'Leafspots Halo', 'Leafspots Marg',
               'Leafspot Size', 'Leaf Shread', 'Leaf Malf', 'Leaf Mild', 'Stem', 'Lodging',
               'Stem Cankers', 'Canker Lesion', 'Fruiting Bodies', 'External Decay', 'Mycelium', 'Interior Discolor',
               'Sclerotia', 'Fruit Pods', 'Fruit Spots', 'Seed', 'Mold Growth', 'Seed Discolor', 'Seed Size',
               'Shriveling', 'Roots', 'Class']
    df.columns = columns

    columns_to_drop = [
        'Date', 'Temp', 'Leafspots Halo', 'Leaf Shread', 'Leaf Malf', 'Leaf Mild',
        'Stem', 'Seed', 'Mold Growth', 'Seed Discolor', 'Seed Size', 'Shriveling'
    ]

    df = df.drop(columns=columns_to_drop)

    class_mapping = {
        'D1': 0,
        'D2': 1,
        'D3': 2,
        'D4': 3
    }

    df['Class'] = df['Class'].map(class_mapping)

    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    num_classes = len(np.unique(y))

    return X, y, num_classes
