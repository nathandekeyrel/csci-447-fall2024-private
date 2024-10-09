import numpy as np
from preprocess import preprocess_data
from knn import KNNClassifier, KNNRegression
from editedKNN import EKNNErrClassifier, EKNNErrRegression
from kMeans import KMeansClassification, KMeansRegression
import tenfoldcv as kfxv
import evaluating as ev
import utils as uti


def demo():
    """
    This is the demo for the video. It shows both a classification example and a regression example.
    Both datasets are loaded. Then, they are preprocessed to form the data in a way where kNN can be used.
    After preprocessing, all 3 variations of the kNN algorithm are performed, with evaluation metrics shown.
    """

    ###################################
    # Load Demo Files
    ###################################

    cancer_filepath = "../data/breast-cancer-wisconsin.data"
    fires_filepath = "../data/forestfires.csv"

    ###################################
    # Classification Example
    ###################################

    # Process Cancer dataset
    print("Processing Cancer dataset:")
    X_cancer, y_cancer = preprocess_data(cancer_filepath)
    X_cancer, y_cancer, cancer_test_X, cancer_test_y = uti.generateTestData(X_cancer, y_cancer)
    print("Preprocessing Complete.")

    # Run raw kNN
    print("Running raw kNN on Cancer dataset...")
    knn_cancer = KNNClassifier()
    knn_cancer.fit(X_cancer, y_cancer)
    knn_results = kfxv.tenfoldcrossvalidationC(knn_cancer, X_cancer, y_cancer, k=1)
    knn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in knn_results])
    print(f"Raw kNN Accuracy: {knn_accuracy:.4f}")

    # Run edited kNN
    # For all edited use the utils test datasets for tenfold cv
    print("Running edited kNN on Cancer dataset...")
    eknn_cancer = EKNNErrClassifier()
    eknn_cancer.fit(X_cancer, y_cancer)
    eknn_cancer.edit(cancer_test_X, cancer_test_y)
    eknn_results = kfxv.tenfoldcrossvalidationC(eknn_cancer, X_cancer, y_cancer, k=1)
    eknn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in eknn_results])
    print(f"Edited kNN Accuracy: {eknn_accuracy:.4f}")

    # Run kMeans kNN
    print("Running kMeans kNN on Cancer dataset...")
    n_clusters = 361  # pulled from edited number of examples returned
    kmeans_cancer = KMeansClassification(n_clusters=n_clusters)
    kmeans_cancer.fit(X_cancer, y_cancer)

    # get the reduced dataset
    rX, rY = kmeans_cancer.get_reduced_dataset()

    knn_cancer = KNNClassifier()
    knn_cancer.fit(rX, rY)

    # cv using the knn classifier on original data
    kmeans_results = kfxv.tenfoldcrossvalidationC(knn_cancer, X_cancer, y_cancer, k=1)
    kmeans_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in kmeans_results])
    print(f"kMeans kNN Accuracy: {kmeans_accuracy:.4f}")

    ###################################
    # Regression Example
    ###################################

    # Process Fires dataset
    print("\nProcessing Fires dataset:")
    X_fires, y_fires = preprocess_data(fires_filepath)
    X_fires, y_fires, fires_test_X, fires_test_y = uti.generateTestData(X_fires, y_fires)
    print("Preprocessing Complete.")

    # Run raw kNN
    print("Running raw kNN on Fires dataset...")
    knn_fires = KNNRegression()
    knn_fires.fit(X_fires, y_fires)
    knn_results = kfxv.tenfoldcrossvalidationR(knn_fires, X_fires, y_fires, k=15, sig=2)
    knn_mse = np.mean([ev.mse(r[0], r[1]) for r in knn_results])
    print(f"Raw kNN MSE: {knn_mse:.4f}")

    # Run edited kNN
    print("Running edited kNN on Fires dataset...")
    eknn_fires = EKNNErrRegression()
    eknn_fires.fit(X_fires, y_fires)
    eknn_fires.edit(fires_test_X, fires_test_y, sig=1, e=28)
    eknn_results = kfxv.tenfoldcrossvalidationR(eknn_fires, X_fires, y_fires, k=13, sig=1, e=5)
    eknn_mse = np.mean([ev.mse(r[0], r[1]) for r in eknn_results])
    print(f"Edited kNN MSE: {eknn_mse:.4f}")

    # Run kMeans kNN
    print("Running kMeans kNN on Fires dataset...")
    n_clusters = 415
    kmeans_fires = KMeansRegression(n_clusters=n_clusters)
    kmeans_fires.fit(X_fires, y_fires)

    rX, rY = kmeans_fires.get_reduced_dataset()

    knn_fires = KNNRegression()
    knn_fires.fit(rX, rY)

    kmeans_results = kfxv.tenfoldcrossvalidationR(knn_fires, X_fires, y_fires, k=13, sig=.5)
    kmeans_mse = np.mean([ev.mse(r[0], r[1]) for r in kmeans_results])
    print(f"kMeans kNN MSE: {kmeans_mse:.4f}")


if __name__ == "__main__":
    demo()
