from preprocess import preprocess_data
from tenfoldcv import train_and_evaluate, summarize_results
import matplotlib.pyplot as plt
import numpy as np


def evaluate_datasets(datasets):
    results = {}
    for name, (X, y) in datasets.items():
        print(f"\n{name} Dataset (original): ")
        results[f'{name}_original'] = train_and_evaluate(X, y)
        print(f"{name} Original Results:")
        summarize_results(results[f'{name}_original'])
    return results


def create_performance_plots(results):
    datasets = ['Cancer', 'Glass', 'Votes', 'Iris', 'Soybean']
    metrics = ['0/1 Loss', 'Precision', 'Recall']

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        data = []

        for dataset in datasets:
            if metric == '0/1 Loss':
                data.append([r[metric] for r in results[f'{dataset}_original']])
            else:
                data.append([np.mean(r[metric]) for r in results[f'{dataset}_original']])

        plt.boxplot(data, labels=datasets)

        plt.title(f'{metric} Across Datasets')
        plt.xlabel('Datasets')
        plt.ylabel(metric)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
        plt.close()


def main():
    # file path for each dataset
    cancer_filepath = '../data/breast-cancer-wisconsin.data'
    glass_filepath = '../data/glass.data'
    votes_filepath = '../data/house-votes-84.data'
    iris_filepath = '../data/iris.data'
    soybeans_filepath = '../data/soybean-small.data'

    # preprocessed split of features and target
    X_cancer, y_cancer = preprocess_data(cancer_filepath)
    X_glass, y_glass = preprocess_data(glass_filepath)
    X_votes, y_votes = preprocess_data(votes_filepath)
    X_iris, y_iris = preprocess_data(iris_filepath)
    X_soybeans, y_soybeans = preprocess_data(soybeans_filepath)

    datasets = {
        'Cancer': (X_cancer, y_cancer),
        'Glass': (X_glass, y_glass),
        'Votes': (X_votes, y_votes),
        'Iris': (X_iris, y_iris),
        'Soybean': (X_soybeans, y_soybeans)
    }

    results = evaluate_datasets(datasets)

    create_performance_plots(results)


if __name__ == '__main__':
    main()
