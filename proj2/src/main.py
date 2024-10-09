import numpy as np
from preprocess import preprocess_data
from knn import KNNClassifier, KNNRegression
from editedKNN import EKNNErrClassifier, EKNNErrRegression
from kMeans import KMeansClassification, KMeansRegression
import tenfoldcv as kfxv
import evaluating as ev
import utils as uti


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
    X_cancer, y_cancer, cancer_test_X, cancer_test_y = uti.generateTestData(X_cancer, y_cancer)

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
    # Running Glass
    ###################################

    # Process Glass dataset
    print("\nProcessing Glass dataset:")
    X_glass, y_glass = preprocess_data(glass_filepath)
    X_glass, y_glass, glass_test_X, glass_test_y = uti.generateTestData(X_glass, y_glass)

    # Run raw kNN
    print("Running raw kNN on Glass dataset...")
    knn_glass = KNNClassifier()
    knn_glass.fit(X_glass, y_glass)
    knn_results = kfxv.tenfoldcrossvalidationC(knn_glass, X_glass, y_glass, k=1)
    knn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in knn_results])
    print(f"Raw kNN Accuracy: {knn_accuracy:.4f}")

    # Run edited kNN
    print("Running edited kNN on Glass dataset...")
    eknn_glass = EKNNErrClassifier()
    eknn_glass.fit(X_glass, y_glass)
    eknn_glass.edit(glass_test_X, glass_test_y)
    eknn_results = kfxv.tenfoldcrossvalidationC(eknn_glass, X_glass, y_glass, k=1)
    eknn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in eknn_results])
    print(f"Edited kNN Accuracy: {eknn_accuracy:.4f}")

    # Run kMeans kNN
    print("Running kMeans kNN on Glass dataset...")
    n_clusters = 59  # pulled from edited number of examples returned
    kmeans_glass = KMeansClassification(n_clusters=n_clusters)
    kmeans_glass.fit(X_glass, y_glass)

    # get the reduced dataset
    rX, rY = kmeans_glass.get_reduced_dataset()

    knn_glass = KNNClassifier()
    knn_glass.fit(rX, rY)

    kmeans_results = kfxv.tenfoldcrossvalidationC(knn_glass, X_glass, y_glass, k=1)
    kmeans_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in kmeans_results])
    print(f"kMeans kNN Accuracy: {kmeans_accuracy:.4f}")

    ###################################
    # Running Soybean
    ###################################

    # Process Soybean dataset
    print("\nProcessing Soybean dataset:")
    X_soybean, y_soybean = preprocess_data(soybean_filepath)
    X_soybean, y_soybean, soybean_test_X, soybean_test_y = uti.generateTestData(X_soybean, y_soybean)

    # Run raw kNN
    print("Running raw kNN on Soybean dataset...")
    knn_soybean = KNNClassifier()
    knn_soybean.fit(X_soybean, y_soybean)
    knn_results = kfxv.tenfoldcrossvalidationC(knn_soybean, X_soybean, y_soybean, k=1)
    knn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in knn_results])
    print(f"Raw kNN Accuracy: {knn_accuracy:.4f}")

    # Run edited kNN
    print("Running edited kNN on Soybean dataset...")
    eknn_soybean = EKNNErrClassifier()
    eknn_soybean.fit(X_soybean, y_soybean)
    eknn_soybean.edit(soybean_test_X, soybean_test_y)
    eknn_results = kfxv.tenfoldcrossvalidationC(eknn_soybean, X_soybean, y_soybean, k=1)
    eknn_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in eknn_results])
    print(f"Edited kNN Accuracy: {eknn_accuracy:.4f}")

    # Run kMeans kNN
    print("Running kMeans kNN on Soybean dataset...")
    n_clusters = 9  # pulled from edited number of examples returned
    kmeans_soybean = KMeansClassification(n_clusters=n_clusters)
    kmeans_soybean.fit(X_soybean, y_soybean)

    rX, rY = kmeans_soybean.get_reduced_dataset()

    knn_soybean = KNNClassifier()
    knn_soybean.fit(rX, rY)

    kmeans_results = kfxv.tenfoldcrossvalidationC(knn_soybean, X_soybean, y_soybean, k=1)
    kmeans_accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in kmeans_results])
    print(f"kMeans kNN Accuracy: {kmeans_accuracy:.4f}")

    ###################################
    # Running Abalone
    ###################################

    # Process Abalone dataset
    print("\nProcessing Abalone dataset:")
    X_abalone, y_abalone = preprocess_data(abalone_filepath)
    X_abalone, y_abalone, abalone_test_X, abalone_test_y = uti.generateTestData(X_abalone, y_abalone)

    # Run raw kNN
    print("Running raw kNN on Abalone dataset...")
    knn_abalone = KNNRegression()
    knn_abalone.fit(X_abalone, y_abalone)
    knn_results = kfxv.tenfoldcrossvalidationR(knn_abalone, X_abalone, y_abalone, k=15, sig=.25)
    knn_mse = np.mean([ev.mse(r[0], r[1]) for r in knn_results])
    print(f"Raw kNN Mse: {knn_mse:.4f}")

    # Run edited kNN
    print("Running edited kNN on Abalone dataset...")
    eknn_abalone = EKNNErrRegression()
    eknn_abalone.fit(X_abalone, y_abalone)
    eknn_abalone.edit(abalone_test_X, abalone_test_y, sig=2, e=21)
    eknn_results = kfxv.tenfoldcrossvalidationR(eknn_abalone, X_abalone, y_abalone, k=15, sig=2, e=21)
    eknn_mse = np.mean([ev.mse(r[0], r[1]) for r in eknn_results])
    print(f"Edited kNN Mse: {eknn_mse:.4f}")

    # Run kMeans kNN
    print("Running kMeans kNN on Abalone dataset...")
    n_clusters = 3379
    kmeans_abalone = KMeansRegression(n_clusters=n_clusters)
    kmeans_abalone.fit(X_abalone, y_abalone)

    rX, rY = kmeans_abalone.get_reduced_dataset()

    knn_abalone = KNNRegression()
    knn_abalone.fit(rX, rY)

    kmeans_results = kfxv.tenfoldcrossvalidationR(knn_abalone, X_abalone, y_abalone, k=15, sig=0.25)
    kmeans_mse = np.mean([ev.mse(r[0], r[1]) for r in kmeans_results])
    print(f"kMeans kNN Mse: {kmeans_mse:.4f}")

    ###################################
    # Running Computer
    ###################################

    # Process Computer dataset
    print("\nProcessing Computer dataset:")
    X_computer, y_computer = preprocess_data(hardware_filepath)
    X_computer, y_computer, computer_test_X, computer_test_y = uti.generateTestData(X_computer, y_computer)

    # Run raw kNN
    print("Running raw kNN on Computer dataset...")
    knn_computer = KNNRegression()
    knn_computer.fit(X_computer, y_computer)
    knn_results = kfxv.tenfoldcrossvalidationR(knn_computer, X_computer, y_computer, k=3, sig=.25)
    knn_mse = np.mean([ev.mse(r[0], r[1]) for r in knn_results])
    print(f"Raw kNN Mse: {knn_mse:.4f}")

    # Run edited kNN
    print("Running edited kNN on Computer dataset...")
    eknn_computer = EKNNErrRegression()
    eknn_computer.fit(X_computer, y_computer)
    eknn_computer.edit(computer_test_X, computer_test_y, sig=.25, e=858)
    eknn_results = kfxv.tenfoldcrossvalidationR(eknn_computer, X_computer, y_computer, k=5, sig=.25, e=858)
    eknn_mse = np.mean([ev.mse(r[0], r[1]) for r in eknn_results])
    print(f"Edited kNN Mse: {eknn_mse:.4f}")

    # Run kMeans kNN
    print("Running kMeans kNN on Computer dataset...")
    n_clusters = 166
    kmeans_computer = KMeansRegression(n_clusters=n_clusters)
    kmeans_computer.fit(X_computer, y_computer)

    rX, rY = kmeans_computer.get_reduced_dataset()

    knn_computer = KNNRegression()
    knn_computer.fit(rX, rY)

    kmeans_results = kfxv.tenfoldcrossvalidationR(knn_computer, X_computer, y_computer, k=1, sig=.25)
    kmeans_mse = np.mean([ev.mse(r[0], r[1]) for r in kmeans_results])
    print(f"kMeans kNN MSE: {kmeans_mse:.4f}")

    ###################################
    # Running Fires
    ###################################

    # Process Fires dataset
    print("\nProcessing Fires dataset:")
    X_fires, y_fires = preprocess_data(fires_filepath)
    X_fires, y_fires, fires_test_X, fires_test_y = uti.generateTestData(X_fires, y_fires)

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
    main()
