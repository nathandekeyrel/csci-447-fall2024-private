from preprocess import preprocess_data


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

    datasets = [
        ("Cancer", cancer_data_original, cancer_data_noisy),
        ("Glass", glass_data_original, glass_data_noisy),
        ("Votes", votes_data_original, votes_data_noisy),
        ("Iris", iris_data_original, iris_data_noisy),
        ("Soybean", soybean_data_original, soybean_data_noisy)
    ]

    print('Raw Num of Instances. Breast: 699, Glass: 214, Votes: 435, Iris: 150, Soybean: 47')

    for name, original, noisy in datasets:
        print(f"\nDataset: {name}")
        print(f"Original shape: {original.shape}")
        print(f"Noisy shape: {noisy.shape}")
        print(f"\nFirst few rows of {name} (original):")
        print(original.head())
        print(f"\nFirst few rows of {name} (noisy):")
        print(noisy.head())
        print("\n" + "=" * 50)


if __name__ == '__main__':
    main()
