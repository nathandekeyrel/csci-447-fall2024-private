from preprocess import preprocess_data
from tenfoldcv import train_and_evaluate, summarize_results
import matplotlib.pyplot as plt
import numpy as np


def main():
    cancer_filepath = '../data/breast-cancer-wisconsin.data'
    glass_filepath = '../data/glass.data'
    votes_filepath = '../data/house-votes-84.data'
    iris_filepath = '../data/iris.data'
    soybean_filepath = '../data/soybean-small.data'

    cancer_data_original, cancer_data_noisy = preprocess_data(cancer_filepath)
    glass_data_original, glass_data_noisy = preprocess_data(glass_filepath)
    votes_data_original, votes_data_noisy = preprocess_data(votes_filepath)
    iris_data_original, iris_data_noisy = preprocess_data(iris_filepath)
    soybean_data_original, soybean_data_noisy = preprocess_data(soybean_filepath)

    results = {}

    # train and eval cancer original
    print("Cancer Dataset (original): ")
    X_cancer, y_cancer = cancer_data_original.iloc[:, 1:].values, cancer_data_original.iloc[:, 0].values
    results['Cancer_original'] = train_and_evaluate(X_cancer, y_cancer)
    print("Cancer Original Results:")
    summarize_results(results['Cancer_original'])

    # train and eval cancer noisy
    print("Cancer Dataset (noisy): ")
    X_cancer_noisy, y_cancer_noisy = cancer_data_noisy.iloc[:, 1:].values, cancer_data_noisy.iloc[:, 0].values
    results['Cancer_noisy'] = train_and_evaluate(X_cancer_noisy, y_cancer_noisy)
    print("Cancer Noisy Results:")
    summarize_results(results['Cancer_noisy'])
    print("-" * 50)

    # train and eval glass original
    print("Glass Dataset (original): ")
    X_glass, y_glass = glass_data_original.iloc[:, 1:].values, glass_data_original.iloc[:, 0].values
    results['Glass_original'] = train_and_evaluate(X_glass, y_glass)
    print("Glass Original Results:")
    summarize_results(results['Glass_original'])

    # train and eval glass noisy
    print("Glass Dataset (noisy): ")
    X_glass_noisy, y_glass_noisy = glass_data_noisy.iloc[:, 1:].values, glass_data_noisy.iloc[:, 0].values
    results['Glass_noisy'] = train_and_evaluate(X_glass_noisy, y_glass_noisy)
    print("Glass Noisy Results:")
    summarize_results(results['Glass_noisy'])
    print("-" * 50)

    # train and eval votes original
    print("Votes Dataset (original): ")
    X_votes, y_votes = votes_data_original.iloc[:, 1:].values, votes_data_original.iloc[:, 0].values
    results['Votes_original'] = train_and_evaluate(X_votes, y_votes)
    print("Votes Original Results:")
    summarize_results(results['Votes_original'])

    # train and eval votes noisy
    print("Votes Dataset (noisy): ")
    X_votes_noisy, y_votes_noisy = votes_data_noisy.iloc[:, 1:].values, votes_data_noisy.iloc[:, 0].values
    results['Votes_noisy'] = train_and_evaluate(X_votes_noisy, y_votes_noisy)
    print("Votes Noisy Results:")
    summarize_results(results['Votes_noisy'])
    print("-" * 50)

    # train and eval iris original
    print("Iris Dataset (original): ")
    X_iris, y_iris = iris_data_original.iloc[:, 1:].values, iris_data_original.iloc[:, 0].values
    results['Iris_original'] = train_and_evaluate(X_iris, y_iris)
    print("Iris Original Results:")
    summarize_results(results['Iris_original'])

    # train and eval iris noisy
    print("Iris Dataset (noisy): ")
    X_iris_noisy, y_iris_noisy = iris_data_noisy.iloc[:, 1:].values, iris_data_noisy.iloc[:, 0].values
    results['Iris_noisy'] = train_and_evaluate(X_iris_noisy, y_iris_noisy)
    print("Iris Noisy Results:")
    summarize_results(results['Iris_noisy'])
    print("-" * 50)

    # train and eval soybean original
    print("Soybean Dataset (original): ")
    X_soybean, y_soybean = soybean_data_original.iloc[:, 1:].values, soybean_data_original.iloc[:, 0].values
    results['Soybean_original'] = train_and_evaluate(X_soybean, y_soybean)
    print("Soybean Original Results:")
    summarize_results(results['Soybean_original'])

    # train and eval soybean noisy
    print("Soybean Dataset (noisy): ")
    X_soybean_noisy, y_soybean_noisy = soybean_data_noisy.iloc[:, 1:].values, soybean_data_noisy.iloc[:, 0].values
    results['Soybean_noisy'] = train_and_evaluate(X_soybean_noisy, y_soybean_noisy)
    print("Soybean Noisy Results:")
    summarize_results(results['Soybean_noisy'])

    datasets = ['Cancer', 'Glass', 'Votes', 'Iris', 'Soybean']
    metrics = ['0/1 Loss', 'Precision', 'Recall']

    for metric in metrics:
        plt.figure(figsize=(12, 6))

        original_values = []
        noisy_values = []

        for d in datasets:
            if metric == '0/1 Loss':
                original_values.append(np.mean([r[metric] for r in results[f'{d}_original']]))
                noisy_values.append(np.mean([r[metric] for r in results[f'{d}_noisy']]))
            else:
                original_values.append(np.mean([np.mean(r[metric]) for r in results[f'{d}_original']]))
                noisy_values.append(np.mean([np.mean(r[metric]) for r in results[f'{d}_noisy']]))

        x = np.arange(len(datasets))
        width = 0.35

        plt.bar(x - width / 2, original_values, width, label='Original')
        plt.bar(x + width / 2, noisy_values, width, label='Noisy')

        plt.xlabel('Datasets')
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} between Original and Noisy Datasets')
        plt.xticks(x, datasets)
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
