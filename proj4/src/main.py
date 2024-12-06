import numpy as np
from preprocess import preprocess_data
import tenfoldcv as kfxc
import evaluating as ev
from GeneticAlgorithm import GeneticAlgorithm
from DifferentialEvolution import DifferentialEvolution
from ParticleSwarmOptimization import PSO


class DatasetConfig:
    def __init__(self, name, filepath, nodes_per_layer):
        self.name = name
        self.filepath = filepath
        self.nodes_per_layer = nodes_per_layer


class ModelConfig:
    def __init__(self, name, model_class, params_by_layer):
        self.name = name
        self.model_class = model_class
        self.params_by_layer = params_by_layer


def load_and_preprocess_data(datasets):
    data_dict = {}
    print("Loading and preprocessing datasets...")
    for dataset in datasets:
        X, y = preprocess_data(dataset.filepath)
        data_dict[dataset.name] = (X, y, dataset.nodes_per_layer)
    return data_dict


def run_experiment(X, y, model_config, n_hidden_layers, n_nodes, is_classifier=True):
    params = model_config.params_by_layer[n_hidden_layers].copy()
    params.update({
        'X_train': X,
        'y_train': y,
        'n_nodes_per_layer': n_nodes,
        'n_hidden_layers': n_hidden_layers,
        'is_classifier': is_classifier
    })

    model = model_config.model_class(**params)
    print(f"\nRunning {model_config.name} with {n_hidden_layers} hidden layer(s)...")

    results = kfxc.tenfoldcrossvalidation(X, y, model)

    if is_classifier:
        accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in results])
        all_true = np.concatenate([r[0] for r in results])
        all_pred = np.concatenate([r[1] for r in results])
        precision = ev.calculate_precision(all_true, all_pred)
        recall = ev.calculate_recall(all_true, all_pred)

        print(f"{n_hidden_layers} Hidden Layers Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision}")
        print(f"  Recall: {recall}")
    else:
        all_true = np.concatenate([r[0] for r in results])
        all_pred = np.concatenate([r[1] for r in results])
        mse_val = ev.mse(all_true, all_pred)
        rmse_val = ev.rmse(all_true, all_pred)
        mae_val = ev.mae(all_true, all_pred)

        print(f"{n_hidden_layers} Hidden Layers Metrics:")
        print(f"  MSE: {mse_val:.4f}")
        print(f"  RMSE: {rmse_val:.4f}")
        print(f"  MAE: {mae_val:.4f}")


def run_all_experiments(data_dict, model_configs, is_classifier=True):
    for dataset_name, (X, y, nodes_per_layer) in data_dict.items():
        print("=" * 75)
        print(f"{dataset_name}:")
        print("=" * 75)

        for model_config in model_configs:
            print(f"\n{model_config.name}:")
            for n_layers in range(3):
                run_experiment(X, y, model_config, n_layers,
                               nodes_per_layer[n_layers], is_classifier)


def main():
    ###################################
    # Classification Dataset Configs
    ###################################
    # Cancer
    cancer_ga_params = {
        0: {'population': 80, 'tournament_size': 3},
        1: {'population': 80, 'tournament_size': 3},
        2: {'population': 80, 'tournament_size': 3}
    }

    cancer_de_params = {
        0: {'population': 48, 'scaling': 1.75, 'binomial_crossover': 0.48},
        1: {'population': 46, 'scaling': 1.84, 'binomial_crossover': 0.45},
        2: {'population': 48, 'scaling': 1.79, 'binomial_crossover': 0.43}
    }

    cancer_pso_params = {
        0: {'population': 35, 'inertia': 0.68, 'cognitive_update_rate': 0.50, 'social_update_rate': 0.091},
        1: {'population': 32, 'inertia': 0.57, 'cognitive_update_rate': 0.47, 'social_update_rate': 0.072},
        2: {'population': 41, 'inertia': 0.61, 'cognitive_update_rate': 0.51, 'social_update_rate': 0.083}
    }

    # Glass
    glass_ga_params = {
        0: {'population': 80, 'tournament_size': 3},
        1: {'population': 30, 'tournament_size': 3},
        2: {'population': 50, 'tournament_size': 3}
    }

    glass_de_params = {
        0: {'population': 25, 'scaling': 1.85, 'binomial_crossover': 0.81},
        1: {'population': 24, 'scaling': 1.97, 'binomial_crossover': 0.80},
        2: {'population': 24, 'scaling': 1.78, 'binomial_crossover': 0.88}
    }

    glass_pso_params = {
        0: {'population': 55, 'inertia': 0.88, 'cognitive_update_rate': 0.55, 'social_update_rate': 0.083},
        1: {'population': 46, 'inertia': 0.92, 'cognitive_update_rate': 0.45, 'social_update_rate': 0.060},
        2: {'population': 50, 'inertia': 0.89, 'cognitive_update_rate': 0.48, 'social_update_rate': 0.12}
    }

    # Soybean
    soybean_ga_params = {
        0: {'population': 80, 'tournament_size': 2},
        1: {'population': 80, 'tournament_size': 3},
        2: {'population': 30, 'tournament_size': 4}
    }

    soybean_de_params = {
        0: {'population': 30, 'scaling': 1.18, 'binomial_crossover': 0.50},
        1: {'population': 32, 'scaling': 1.14, 'binomial_crossover': 0.47},
        2: {'population': 32, 'scaling': 1.12, 'binomial_crossover': 0.48}
    }

    soybean_pso_params = {
        0: {'population': 57, 'inertia': 0.53, 'cognitive_update_rate': 0.40, 'social_update_rate': 0.28},
        1: {'population': 46, 'inertia': 0.20, 'cognitive_update_rate': 0.71, 'social_update_rate': 0.51},
        2: {'population': 50, 'inertia': 0.031, 'cognitive_update_rate': 0.74, 'social_update_rate': 0.22}
    }

    ###################################
    # Regression Dataset Configs
    ##################################
    # Abalone
    abalone_ga_params = {
        0: {'population': 80, 'tournament_size': 4},
        1: {'population': 50, 'tournament_size': 3},
        2: {'population': 80, 'tournament_size': 4}
    }

    abalone_de_params = {
        0: {'population': 30, 'scaling': 0.45, 'binomial_crossover': 0.40},
        1: {'population': 30, 'scaling': 0.45, 'binomial_crossover': 0.40},
        2: {'population': 22, 'scaling': 1.25, 'binomial_crossover': 0.18}
    }

    abalone_pso_params = {
        0: {'population': 27, 'inertia': 0.85, 'cognitive_update_rate': 0.80, 'social_update_rate': 0.87},
        1: {'population': 51, 'inertia': 0.92, 'cognitive_update_rate': 0.75, 'social_update_rate': 0.51},
        2: {'population': 21, 'inertia': 0.95, 'cognitive_update_rate': 0.11, 'social_update_rate': 0.67}
    }

    # Hardware
    machine_ga_params = {
        0: {'population': 50, 'tournament_size': 4},
        1: {'population': 50, 'tournament_size': 3},
        2: {'population': 80, 'tournament_size': 4}
    }

    machine_de_params = {
        0: {'population': 54, 'scaling': 1.25, 'binomial_crossover': 0.83},
        1: {'population': 50, 'scaling': 1.32, 'binomial_crossover': 0.95},
        2: {'population': 49, 'scaling': 1.32, 'binomial_crossover': 0.89}
    }

    machine_pso_params = {
        0: {'population': 26, 'inertia': 0.92, 'cognitive_update_rate': 0.095, 'social_update_rate': 0.52},
        1: {'population': 26, 'inertia': 0.92, 'cognitive_update_rate': 0.095, 'social_update_rate': 0.52},
        2: {'population': 29, 'inertia': 0.90, 'cognitive_update_rate': 0.92, 'social_update_rate': 0.74}
    }

    # Fires
    fires_ga_params = {
        0: {'population': 50, 'tournament_size': 3},
        1: {'population': 30, 'tournament_size': 4},
        2: {'population': 80, 'tournament_size': 4}
    }

    fires_de_params = {
        0: {'population': 31, 'scaling': 1.72, 'binomial_crossover': 0.63},
        1: {'population': 31, 'scaling': 1.77, 'binomial_crossover': 0.67},
        2: {'population': 33, 'scaling': 1.81, 'binomial_crossover': 0.67}
    }

    fires_pso_params = {
        0: {'population': 53, 'inertia': 0.81, 'cognitive_update_rate': 0.85, 'social_update_rate': 0.74},
        1: {'population': 50, 'inertia': 0.96, 'cognitive_update_rate': 0.78, 'social_update_rate': 0.83},
        2: {'population': 55, 'inertia': 0.90, 'cognitive_update_rate': 0.81, 'social_update_rate': 0.80}
    }

    ###################################
    # Classification Datasets
    ###################################
    classification_datasets = [
        DatasetConfig("Cancer", "../data/breast-cancer-wisconsin.data", [5, 5, 5]),
        DatasetConfig("Glass", "../data/glass.data", [7, 7, 7]),
        DatasetConfig("Soybean", "../data/soybean-small.data", [38, 38, 38])
    ]

    ###################################
    # Regression Datasets
    ###################################
    regression_datasets = [
        DatasetConfig("Abalone", "../data/abalone.data", [13, 13, 12]),
        DatasetConfig("Hardware", "../data/machine.data", [30, 28, 64]),
        DatasetConfig("Fires", "../data/forestfires.csv", [54, 25, 35])
    ]

    ###################################
    # Run Classification
    ###################################
    print("\nRunning Classification Experiments...")
    classification_data = load_and_preprocess_data(classification_datasets)

    # Cancer
    cancer_models = [
        ModelConfig("GENETIC ALGORITHM", GeneticAlgorithm, cancer_ga_params),
        ModelConfig("DIFFERENTIAL EVOLUTION", DifferentialEvolution, cancer_de_params),
        ModelConfig("PARTICLE SWARM OPTIMIZATION", PSO, cancer_pso_params)
    ]
    run_all_experiments({"Cancer": classification_data["Cancer"]}, cancer_models, True)

    # Glass
    glass_models = [
        ModelConfig("GENETIC ALGORITHM", GeneticAlgorithm, glass_ga_params),
        ModelConfig("DIFFERENTIAL EVOLUTION", DifferentialEvolution, glass_de_params),
        ModelConfig("PARTICLE SWARM OPTIMIZATION", PSO, glass_pso_params)
    ]
    run_all_experiments({"Glass": classification_data["Glass"]}, glass_models, True)

    # Soybean
    soybean_models = [
        ModelConfig("GENETIC ALGORITHM", GeneticAlgorithm, soybean_ga_params),
        ModelConfig("DIFFERENTIAL EVOLUTION", DifferentialEvolution, soybean_de_params),
        ModelConfig("PARTICLE SWARM OPTIMIZATION", PSO, soybean_pso_params)
    ]
    run_all_experiments({"Soybean": classification_data["Soybean"]}, soybean_models, True)

    ###################################
    # Run Regression
    ###################################
    print("\nRunning Regression Experiments...")
    regression_data = load_and_preprocess_data(regression_datasets)

    # Abalone
    abalone_models = [
        ModelConfig("GENETIC ALGORITHM", GeneticAlgorithm, abalone_ga_params),
        ModelConfig("DIFFERENTIAL EVOLUTION", DifferentialEvolution, abalone_de_params),
        ModelConfig("PARTICLE SWARM OPTIMIZATION", PSO, abalone_pso_params)
    ]
    run_all_experiments({"Abalone": regression_data["Abalone"]}, abalone_models, False)

    # Machine
    machine_models = [
        ModelConfig("GENETIC ALGORITHM", GeneticAlgorithm, machine_ga_params),
        ModelConfig("DIFFERENTIAL EVOLUTION", DifferentialEvolution, machine_de_params),
        ModelConfig("PARTICLE SWARM OPTIMIZATION", PSO, machine_pso_params)
    ]
    run_all_experiments({"Hardware": regression_data["Hardware"]}, machine_models, False)

    # Fires
    fires_models = [
        ModelConfig("GENETIC ALGORITHM", GeneticAlgorithm, fires_ga_params),
        ModelConfig("DIFFERENTIAL EVOLUTION", DifferentialEvolution, fires_de_params),
        ModelConfig("PARTICLE SWARM OPTIMIZATION", PSO, fires_pso_params)
    ]
    run_all_experiments({"Fires": regression_data["Fires"]}, fires_models, False)


if __name__ == "__main__":
    main()
