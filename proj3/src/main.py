from preprocess import preprocess_data


def main():
    ###################################
    # Load datasets
    ###################################
    cancer_filepath = "../data/breast-cancer-wisconsin.data"
    glass_filepath = "../data/glass.data"
    soybean_filepath = "../data/soybean-small.data"
    abalone_filepath = "../data/abalone.data"
    hardware_filepath = "../data/machine.data"
    fires_filepath = "../data/forestfires.csv"

    ###################################
    # Load datasets
    ###################################

    X_cancer, y_cancer = preprocess_data(cancer_filepath)
    X_glass, y_glass = preprocess_data(glass_filepath)
    X_soybean, y_soybean = preprocess_data(soybean_filepath)
    X_abalone, y_abalone = preprocess_data(abalone_filepath)
    X_hardware, y_hardware = preprocess_data(hardware_filepath)
    X_fires, y_fires = preprocess_data(fires_filepath)


if __name__ == "__main__":
    main()
