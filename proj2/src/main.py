from preprocess import preprocess_data
from knn import KNNClassifier, KNNRegression
from editedKNN import EKNNErrClassifier, EKNNErrRegression
from kMeans import KMeans

def display_results():
    pass


def main():
    ###################################
    # Load datasets
    ###################################

    # store filepaths for each dataset
    cancer_filepath = "../data/breast-cancer-wisconsin.data"
    glass_filepath = "../data/glass.data"
    soybean_filepath = "../data/soybean-small.data"
    abalone_filepath = "../data/abalone.data"
    hardware_filepath = "../data/machine.data"
    fires_filepath = "../data/forestfires.csv"

    ###################################
    # Preprocess data
    ###################################

    # preprocess breast-cancer
    X_cancer, y_cancer = preprocess_data(cancer_filepath)

    # preprocess glass
    X_glass, y_glass = preprocess_data(glass_filepath)

    # preprocess soybean
    X_soybean, y_soybean = preprocess_data(soybean_filepath)

    # preprocess abalone
    X_abalone, y_abalone = preprocess_data(abalone_filepath)

    # preprocess computer hardware
    X_hardware, y_hardware = preprocess_data(hardware_filepath)

    # preprocess forest fires
    X_fires, y_fires = preprocess_data(fires_filepath)

    ###################################
    # Run algorithms on each dataset
    ###################################

    # regular KNN

    # edited KNN

    # kMeans KNN
    n_clusters = 5  # placeholder, will be determined by either tuning or eKNN
    knn_k = 3  # placeholder k for KNN, will be determined later

    # perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_cancer, y_cancer)

    # get reduced dataset
    cancer_reduced_X, cancer_reduced_y = kmeans.get_reduced_dataset()
    # train KNN on the reduced dataset
    cancer_knn = KNNClassifier()
    cancer_knn.fit(cancer_reduced_X, cancer_reduced_y)

    # predict on the original dataset
    cancer_test_clusters = kmeans.predict(X_cancer)
    cancer_predictions = cancer_knn.predict(kmeans.centroids[cancer_test_clusters], k=knn_k)

    print(cancer_predictions)

    """ 
    for demo purposes, we will just run in the following manner:
    
    regular KNN
    edited KNN
    kMeans KNN
    
    This will also allow us to properly call classification or regression depending on the set
    I will show both classification and regression in the demo. We need to select datasets for that
    """


if __name__ == "__main__":
    main()
