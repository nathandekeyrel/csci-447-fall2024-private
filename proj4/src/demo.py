import numpy as np
from preprocess import preprocess_data
import tenfoldcv as kfxc
import evaluating as ev
from GeneticAlgorithm import GeneticAlgorithm
from DifferentialEvolution import DifferentialEvolution
from ParticleSwarmOptimization import PSO


##############################################
# fix
##############################################
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
            for n_layers in [2]:
                run_experiment(X, y, model_config, n_layers,
                               nodes_per_layer[n_layers], is_classifier)


def demo():
    ###################################
    # Classification Dataset Configs
    ###################################

    # Soybean
    soybean_ga_params = {
        2: {'population': 30, 'tournament_size': 4}
    }

    soybean_de_params = {
        2: {'population': 32, 'scaling': 1.12, 'binomial_crossover': 0.48}
    }

    soybean_pso_params = {
        2: {'population': 50, 'inertia': 0.031, 'cognitive_update_rate': 0.74, 'social_update_rate': 0.22}
    }

    ###################################
    # Classification Datasets
    ###################################
    classification_datasets = [
        DatasetConfig("Soybean", "../data/soybean-small.data", [38, 38, 38])
    ]

    ###################################
    # Run Classification
    ###################################
    print("\nRunning Classification Experiments...")
    classification_data = load_and_preprocess_data(classification_datasets)

    # Soybean
    soybean_models = [
        ModelConfig("GENETIC ALGORITHM", GeneticAlgorithm, soybean_ga_params),
        ModelConfig("DIFFERENTIAL EVOLUTION", DifferentialEvolution, soybean_de_params),
        ModelConfig("PARTICLE SWARM OPTIMIZATION", PSO, soybean_pso_params)
    ]
    run_all_experiments({"Soybean": classification_data["Soybean"]}, soybean_models, True)

    ###################################
    # Regression Dataset Configs
    ##################################
    # Fires
    fires_ga_params = {
        2: {'population': 80, 'tournament_size': 4}
    }

    fires_de_params = {
        2: {'population': 33, 'scaling': 1.81, 'binomial_crossover': 0.67}
    }

    fires_pso_params = {
        2: {'population': 55, 'inertia': 0.90, 'cognitive_update_rate': 0.81, 'social_update_rate': 0.80}
    }

    ###################################
    # Regression Datasets
    ###################################
    regression_datasets = [
        DatasetConfig("Fires", "../data/forestfires.csv", [54, 25, 35])
    ]

    ###################################
    # Run Regression
    ###################################
    print("\nRunning Regression Experiments...")
    regression_data = load_and_preprocess_data(regression_datasets)

    # Fires
    fires_models = [
        ModelConfig("GENETIC ALGORITHM", GeneticAlgorithm, fires_ga_params),
        ModelConfig("DIFFERENTIAL EVOLUTION", DifferentialEvolution, fires_de_params),
        ModelConfig("PARTICLE SWARM OPTIMIZATION", PSO, fires_pso_params)
    ]
    run_all_experiments({"Fires": regression_data["Fires"]}, fires_models, False)


if __name__ == "__main__":
    demo()
