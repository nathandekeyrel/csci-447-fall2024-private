import pandas as pd


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
    """Preprocess Cancer dataset...

    :param filepath:
    :return:
    """
    df = pd.read_csv(filepath, header=None, na_values='?')

    columns = ['Id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
               'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
               'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    df.columns = columns
    df = df.dropna()

    df = df.drop('Id', axis=1)

    # Target is 'Class'


def _preprocess_glass(filepath):
    """Preprocess Glass dataset...

    :param filepath:
    :return:
    """
    df = pd.read_csv(filepath, header=None, na_values='?')

    columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    df.columns = columns

    df = df.drop('Id', axis=1)

    # Target is 'Type'


def _preprocess_soybean(filepath):
    """Preprocess Soybean dataset...

    :param filepath:
    :return:
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

    # Target: 'Class'. may need to map


def _preprocess_abalone(filepath):
    """Preprocess Abalone dataset...

    :param filepath:
    :return:
    """
    df = pd.read_csv(filepath, header=None)

    columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight',
               'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
    df.columns = columns

    # all relevant attributes
    # no missing values

    # Target: 'Rings' --> Age = Rings + 1.5


def _preprocess_fires(filepath):
    """Preprocess Forest Fires dataset...

    :param filepath:
    :return:
    """
    df = pd.read_csv(filepath)

    columns = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC',
               'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
    df.columns = columns

    # all attributes seem relevant.
    # no missing values

    # Note: several of the attributes may be correlated, thus it makes sense to apply some sort of feature selection.

    # Target: area - the burned area of the forest (in ha): 0.00 to 1090.84
    #                (this output variable is very skewed towards 0.0, thus it may make
    #                sense to model with the logarithm transform)


def _preprocess_computer(filepath):
    """Preprocess Computer dataset...

    :param filepath:
    :return:
    """
    df = pd.read_csv(filepath, header=None)

    columns = ['Vendor Name', 'Model Name', 'MYCT', 'MMIN',
               'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
    df.columns = columns

    # can probably drop vendor name and model name. unlikely to add anything. Check 'ERP'
    # no missing values

    # Target: 'PRP', continuously valued
