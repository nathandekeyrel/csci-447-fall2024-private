import numpy as np
import pandas as pd


def preprocess_data(filepath):
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
        raise ValueError(f"Unknown dataset: {filepath}")


def preprocess_cancer(filepath):
    # Read the data
    df = pd.read_csv(filepath, header=None, na_values='?')

    # Assign column names
    columns = ['Id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
               'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
               'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    df.columns = columns

    df = df.dropna()

    df = df.drop('Id', axis=1)

    for col in df.columns[:-1]:
        df[col] = pd.cut(df[col], bins=3, labels=[0, 1, 2])

    df['Class'] = df['Class'].map({2: 0, 4: 1})

    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    return X, y


def preprocess_glass(filepath):
    df = pd.read_csv(filepath, header=None, na_values='?')

    columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    df.columns = columns

    df = df.drop('Id', axis=1)

    X = df.drop('Type', axis=1)
    y = df['Type']

    for column in X.columns:
        try:
            X[column] = pd.qcut(X[column], q=4, labels=False, duplicates='drop')
        except ValueError:
            X[column] = pd.cut(X[column], bins=4, labels=False, include_lowest=True)

    X = X.values
    y = y.values

    unique_classes = np.unique(y)
    class_map = {c: i for i, c in enumerate(unique_classes)}
    y = np.array([class_map[c] for c in y])

    return X, y

def preprocess_votes(filepath):
    df = pd.read_csv(filepath, header=None)

    columns = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing',
               'adoption-of-the-budget-resolution', 'physician-fee-freeze',
               'el-salvador-aid', 'religious-groups-in-schools',
               'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
               'mx-missile', 'immigration', 'synfuels-corporation-cutback',
               'education-spending', 'superfund-right-to-sue', 'crime',
               'duty-free-exports', 'export-administration-act-south-africa']
    df.columns = columns

    # Convert 'y' to 1, 'n' to 0, and '?' to NaN
    df = df.replace({'y': 1, 'n': 0, '?': 2})

    df['Class Name'] = df['Class Name'].map({'democrat': 0, 'republican': 1})

    X = df.drop('Class Name', axis=1).values
    y = df['Class Name'].values

    return X, y


def preprocess_iris(filepath):
    df = pd.read_csv(filepath, header=None)

    columns = ['Sepal Width', 'Sepal Length', 'Petal Length', 'Petal Width', 'Class']
    df.columns = columns

    for column in df.columns[:-1]:
        df[column] = pd.qcut(df[column], q=4, labels=False, duplicates='drop')

    class_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

    df['Class'] = df['Class'].map(class_mapping)

    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    return X, y


def preprocess_soybean(filepath):
    df = pd.read_csv(filepath, header=None)

    columns = ['Date', 'Plant Stand', 'Precip', 'Temp', 'Hail', 'Crop-hist', 'Area Damaged', 'Severity',
               'Seed Tmt', 'Germination', 'Plant Growth', 'Leaves', 'Leafspots Halo', 'Leafspots Marg',
               'Leafspot Size', 'Leaf Shread', 'Leaf Malf', 'Leaf Mild', 'Stem', 'Lodging',
               'Stem Cankers', 'Canker Lesion', 'Fruiting Bodies', 'External Decay', 'Mycelium', 'Interior Discolor',
               'Sclerotia', 'Fruit Pods', 'Fruit Spots', 'Seed', 'Mold Growth', 'Seed Discolor', 'Seed Size',
               'Shriveling', 'Roots', 'Class']
    df.columns = columns

    df['Class'] = df['Class'].map({'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4})

    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    return X, y
