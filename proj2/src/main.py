import numpy as np
from preprocess import preprocess_data
from knn import KNNClassifier, KNNRegression
from editedKNN import EKNNErrClassifier, EKNNErrRegression
from kMeans import KMeansClassification, KMeansRegression
import tenfoldcv as kfxv
import evaluating as ev


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
    # Running Cancer
    ###################################

    # Process Cancer dataset
    print("Processing Cancer dataset:")
    X_cancer, y_cancer = preprocess_data(cancer_filepath)

    # Run raw kNN
    print("Running raw kNN on Cancer dataset...")
    knn_cancer = KNNClassifier()
    knn_cancer.fit(X_cancer, y_cancer)
    knn_results = kfxv.tenfoldcrossvalidationC(knn_cancer, X_cancer, y_cancer, k=10)
    knn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in knn_results])
    print(f"Raw kNN Accuracy: {knn_accuracy:.4f}")

    # Run edited kNN
    print("Running edited kNN on Cancer dataset...")
    eknn_cancer = EKNNErrClassifier()
    eknn_cancer.fit(X_cancer, y_cancer)
    eknn_cancer.edit(X_cancer, y_cancer)
    eknn_results = kfxv.tenfoldcrossvalidationC(eknn_cancer, X_cancer, y_cancer, k=10)
    eknn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in eknn_results])
    print(f"Edited kNN Accuracy: {eknn_accuracy:.4f}")

    # Run kMeans kNN
    print("Running kMeans kNN on Cancer dataset...")
    n_clusters = eknn_cancer.numberOfExamples()
    kmeans_cancer = KMeansClassification(n_clusters=n_clusters)
    kmeans_cancer.fit(X_cancer, y_cancer)

    # get the reduced dataset
    rX, rY = kmeans_cancer.get_reduced_dataset()

    knn_cancer = KNNClassifier()
    knn_cancer.fit(rX, rY)

    # cv using the knn classifier on original data
    kmeans_results = kfxv.tenfoldcrossvalidationC(knn_cancer, X_cancer, y_cancer, k=10)
    kmeans_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in kmeans_results])
    print(f"kMeans kNN Accuracy: {kmeans_accuracy:.4f}")
    #
    # ###################################
    # # Running Glass
    # ###################################
    #
    # # Process Glass dataset
    # print("\nProcessing Glass dataset:")
    # X_glass, y_glass = preprocess_data(glass_filepath)
    #
    # # Run raw kNN
    # print("Running raw kNN on Glass dataset...")
    # knn_glass = KNNClassifier()
    # knn_glass.fit(X_glass, y_glass)
    # knn_results = kfxv.tenfoldcrossvalidationC(knn_glass, X_glass, y_glass, k=10)
    # knn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in knn_results])
    # print(f"Raw kNN Accuracy: {knn_accuracy:.4f}")
    #
    # # Run edited kNN
    # print("Running edited kNN on Glass dataset...")
    # eknn_glass = EKNNErrClassifier()
    # eknn_glass.fit(X_glass, y_glass)
    # eknn_glass.edit(X_glass, y_glass)
    # eknn_results = kfxv.tenfoldcrossvalidationC(eknn_glass, X_glass, y_glass, k=10)
    # eknn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in eknn_results])
    # print(f"Edited kNN Accuracy: {eknn_accuracy:.4f}")
    #
    # # Run kMeans kNN
    # print("Running kMeans kNN on Glass dataset...")
    # n_clusters = eknn_glass.numberOfExamples()
    # kmeans_glass = KMeansClassification(n_clusters=n_clusters)
    # kmeans_glass.fit(X_glass, y_glass)
    #
    # # get the reduced dataset
    # rX, rY = kmeans_glass.get_reduced_dataset()
    #
    # knn_glass = KNNClassifier()
    # knn_glass.fit(rX, rY)
    #
    # kmeans_results = kfxv.tenfoldcrossvalidationC(kmeans_glass, X_glass, y_glass, k=10)
    # kmeans_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in kmeans_results])
    # print(f"kMeans kNN Accuracy: {kmeans_accuracy:.4f}")
    #
    # ###################################
    # # Running Soybean
    # ###################################
    #
    # # Process Soybean dataset
    # print("\nProcessing Soybean dataset:")
    # X_soybean, y_soybean = preprocess_data(soybean_filepath)
    #
    # # Run raw kNN
    # print("Running raw kNN on Soybean dataset...")
    # knn_soybean = KNNClassifier()
    # knn_soybean.fit(X_soybean, y_soybean)
    # knn_results = kfxv.tenfoldcrossvalidationC(knn_soybean, X_soybean, y_soybean, k=10)
    # knn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in knn_results])
    # print(f"Raw kNN Accuracy: {knn_accuracy:.4f}")
    #
    # # Run edited kNN
    # print("Running edited kNN on Soybean dataset...")
    # eknn_soybean = EKNNErrClassifier()
    # eknn_soybean.fit(X_soybean, y_soybean)
    # eknn_soybean.edit(X_soybean, y_soybean)
    # eknn_results = kfxv.tenfoldcrossvalidationC(eknn_soybean, X_soybean, y_soybean, k=10)
    # eknn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in eknn_results])
    # print(f"Edited kNN Accuracy: {eknn_accuracy:.4f}")
    #
    # # Run kMeans kNN
    # print("Running kMeans kNN on Soybean dataset...")
    # n_clusters = eknn_soybean.numberOfExamples()
    # kmeans_soybean = KMeansClassification(n_clusters=n_clusters)
    # kmeans_soybean.fit(X_soybean, y_soybean)
    #
    # rX, rY = kmeans_soybean.get_reduced_dataset()
    #
    # knn_soybean = KNNClassifier()
    # knn_soybean.fit(rX, rY)
    #
    # kmeans_results = kfxv.tenfoldcrossvalidationC(kmeans_soybean, X_soybean, y_soybean, k=10)
    # kmeans_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in kmeans_results])
    # print(f"kMeans kNN Accuracy: {kmeans_accuracy:.4f}")
    #
    # # Process regression datasets (Abalone, Computer Hardware, Forest Fires)
    # # ... (implement similar process using KNNRegression, EKNNErrRegression, and KMeansRegression)
    #
    # ###################################
    # # Running Abalone
    # ###################################
    #
    # # Process Abalone dataset
    # print("\nProcessing Abalone dataset:")
    # X_abalone, y_abalone = preprocess_data(abalone_filepath)
    #
    # # Run raw kNN
    # print("Running raw kNN on Abalone dataset...")
    # knn_abalone = KNNRegression()
    # knn_abalone.fit(X_abalone, y_abalone)
    # knn_results = kfxv.tenfoldcrossvalidationC(knn_abalone, X_abalone, y_abalone, k=10)
    # knn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in knn_results])
    # print(f"Raw kNN Accuracy: {knn_accuracy:.4f}")
    #
    # # Run edited kNN
    # print("Running edited kNN on Abalone dataset...")
    # eknn_abalone = EKNNErrRegression()
    # eknn_abalone.fit(X_abalone, y_abalone)
    # eknn_abalone.edit(X_abalone, y_abalone, sig=.125, e=28)
    # eknn_results = kfxv.tenfoldcrossvalidationC(eknn_abalone, X_abalone, y_abalone, k=10)
    # eknn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in eknn_results])
    # print(f"Edited kNN Accuracy: {eknn_accuracy:.4f}")
    #
    # # Run kMeans kNN
    # print("Running kMeans kNN on Abalone dataset...")
    # n_clusters = eknn_abalone.numberOfExamples()
    # kmeans_abalone = KMeansRegression(n_clusters=n_clusters)
    # kmeans_abalone.fit(X_abalone, y_abalone)
    #
    # rX, rY = kmeans_abalone.get_reduced_dataset()
    #
    # knn_abalone = KNNClassifier()
    # knn_abalone.fit(rX, rY)
    #
    # kmeans_results = kfxv.tenfoldcrossvalidationC(kmeans_abalone, X_abalone, y_abalone, k=10)
    # kmeans_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in kmeans_results])
    # print(f"kMeans kNN Accuracy: {kmeans_accuracy:.4f}")


if __name__ == "__main__":
    main()
