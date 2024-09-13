import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from tenfoldcv import train_and_evaluate, summarize_results


def create_noise(X):
    """
    Add noise to input data by shuffling 10% of its features.

    :param X: Input data array
    :return: Copy of input data with noise added
    """
    X_noisy = X.copy()
    num_features = X.shape[1]
    num_of_noise_features = max(1, math.floor(num_features * 0.1))
    noise_features = random.sample(range(num_features), num_of_noise_features)

    for feature in noise_features:
        np.random.shuffle(X_noisy[:, feature])

    return X_noisy


def evaluate_datasets(datasets, is_noisy=False):
    """
    Evaluate multiple datasets, optionally adding noise.

    :param datasets: Dictionary of datasets (name: (X, y, num_classes))
    :param is_noisy: If True, add noise to datasets before evaluation
    :return: Dictionary of evaluation results for each dataset
    """
    results = {}
    for name, (X, y, num_classes) in datasets.items():
        if is_noisy:
            X = create_noise(X)
            print(f"\n{name} Dataset (noisy): ")
            key = f'{name}_noisy'
        else:
            print(f"\n{name} Dataset (original): ")
            key = f'{name}_original'

        fold_results, summed_cm = train_and_evaluate(X, y, num_classes)
        results[key] = (fold_results, summed_cm)
        print(f"{name} {'Noisy' if is_noisy else 'Original'} Results:")
        summarize_results(fold_results, summed_cm)
    return results


def create_performance_plots(results):
    """
    Create box plots comparing original and noisy dataset performance.

    :param results: Dictionary containing evaluation results for datasets
    """
    datasets = ['Cancer', 'Glass', 'Votes', 'Iris', 'Soybean']
    metrics = ['0/1 Loss', 'Precision', 'Recall', 'F1 Score']

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        data_original = []
        data_noisy = []

        for dataset in datasets:
            original_results = results[f'{dataset}_original'][0]
            noisy_results = results[f'{dataset}_noisy'][0]

            if metric == '0/1 Loss':
                data_original.append([r[metric] for r in original_results])
                data_noisy.append([r[metric] for r in noisy_results])
            else:
                data_original.append([np.mean(r[metric]) for r in original_results])
                data_noisy.append([np.mean(r[metric]) for r in noisy_results])

        positions = range(1, len(datasets) * 2 + 1, 2)
        box_original = plt.boxplot(data_original, positions=positions,
                                   labels=[f"{d}" for d in datasets], patch_artist=True)
        box_noisy = plt.boxplot(data_noisy, positions=[p + 1 for p in positions],
                                labels=[f"{d}" for d in datasets], patch_artist=True)

        for box in box_original['boxes']:
            box.set_facecolor('lightblue')
        for box in box_noisy['boxes']:
            box.set_facecolor('lightgreen')

        plt.title(f'{metric} Across Datasets (Original vs Noisy)')
        plt.xlabel('Datasets')
        plt.ylabel(metric)

        plt.legend([box_original["boxes"][0], box_noisy["boxes"][0]], ['Original', 'Noisy'], loc='best')

        plt.tight_layout()

        filename = os.path.join('../images', f'{metric.lower().replace("/", "_")}_boxplot.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()


def main():
    # file path for each dataset
    cancer_filepath = '../data/breast-cancer-wisconsin.data'
    glass_filepath = '../data/glass.data'
    votes_filepath = '../data/house-votes-84.data'
    iris_filepath = '../data/iris.data'
    soybeans_filepath = '../data/soybean-small.data'

    # preprocessed split of features, target, and number of unique classes in target
    X_cancer, y_cancer, num_classes_cancer = preprocess_data(cancer_filepath)
    X_glass, y_glass, num_classes_glass = preprocess_data(glass_filepath)
    X_votes, y_votes, num_classes_votes = preprocess_data(votes_filepath)
    X_iris, y_iris, num_classes_iris = preprocess_data(iris_filepath)
    X_soybeans, y_soybeans, num_classes_soybean = preprocess_data(soybeans_filepath)

    # dictionary of datasets
    datasets = {
        'Cancer': (X_cancer, y_cancer, num_classes_cancer),
        'Glass': (X_glass, y_glass, num_classes_glass),
        'Votes': (X_votes, y_votes, num_classes_votes),
        'Iris': (X_iris, y_iris, num_classes_iris),
        'Soybean': (X_soybeans, y_soybeans, num_classes_soybean)
    }

    # eval original and noisy
    results_original = evaluate_datasets(datasets, is_noisy=False)
    results_noisy = evaluate_datasets(datasets, is_noisy=True)

    """
    creates new dictionary that contains all the keys 
    in original and noisy using dictionary unpacking
    references:
        - https://discuss.python.org/t/syntax-for-dictionnary-unpacking-to-variables/18718
        - https://medium.com/@ashishkush1122/dictionary-unpacking-in-python-544f957e035a
    """
    results = {**results_original, **results_noisy}

    # generate plots
    create_performance_plots(results)


if __name__ == '__main__':
    main()
