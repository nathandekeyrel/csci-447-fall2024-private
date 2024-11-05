import numpy as np
from preprocess import preprocess_data
import tenfoldcv as kfxc
import evaluating as ev


def demo():
    ###################################
    # Load datasets
    ###################################
    print("Loading data...")
    cancer_filepath = "../data/breast-cancer-wisconsin.data"
    fires_filepath = "../data/forestfires.csv"

    ###################################
    # Preprocess Data
    ###################################
    print("Preprocessing datasets...")
    X_cancer, y_cancer = preprocess_data(cancer_filepath)
    X_fires, y_fires = preprocess_data(fires_filepath)

    ###################################
    # Run Cancer
    ###################################
    print("=" * 75)
    print("Cancer:")
    print("=" * 75)

    # 0 hidden layers
    print("Running 0 hidden layers...")
    results = kfxc.tenfoldcrossvalidationC(X_cancer, y_cancer,
                                           hidden_layers=0,
                                           nodes_per_hidden_layer=5,
                                           batch_size=8,
                                           learning_rate=0.9,
                                           momentum=0.9)

    accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in results])
    all_true = np.concatenate([r[0] for r in results])
    all_pred = np.concatenate([r[1] for r in results])
    precision = ev.calculate_precision(all_true, all_pred)
    recall = ev.calculate_recall(all_true, all_pred)

    print(f"0 Hidden Layers Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")

    # 1 hidden layer
    print("\nRunning 1 hidden layer...")
    results = kfxc.tenfoldcrossvalidationC(X_cancer, y_cancer,
                                           hidden_layers=1,
                                           nodes_per_hidden_layer=5,
                                           batch_size=4,
                                           learning_rate=0.3,
                                           momentum=0.7)

    accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in results])
    all_true = np.concatenate([r[0] for r in results])
    all_pred = np.concatenate([r[1] for r in results])
    precision = ev.calculate_precision(all_true, all_pred)
    recall = ev.calculate_recall(all_true, all_pred)

    print(f"1 Hidden Layer Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")

    # 2 hidden layers
    print("\nRunning 2 hidden layers...")
    results = kfxc.tenfoldcrossvalidationC(X_cancer, y_cancer,
                                           hidden_layers=2,
                                           nodes_per_hidden_layer=5,
                                           batch_size=16,
                                           learning_rate=0.5,
                                           momentum=0.9)

    accuracy = 1 - np.mean([ev.zero_one_loss(r[0], r[1]) for r in results])
    all_true = np.concatenate([r[0] for r in results])
    all_pred = np.concatenate([r[1] for r in results])
    precision = ev.calculate_precision(all_true, all_pred)
    recall = ev.calculate_recall(all_true, all_pred)
    print(f"2 Hidden Layers Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")

    ###################################
    # Running Fires
    ###################################
    print("=" * 75)
    print("Fires:")
    print("=" * 75)

    # 0 hidden layers
    print("Running 0 hidden layers...")
    results = kfxc.tenfoldcrossvalidationR(X_fires, y_fires,
                                           hidden_layers=0,
                                           nodes_per_hidden_layer=54,
                                           batch_size=169,
                                           learning_rate=0.617997,
                                           momentum=0.479931)

    all_true = np.concatenate([r[0] for r in results])
    all_pred = np.concatenate([r[1] for r in results])
    mse_score = ev.mse(all_true, all_pred)
    rmse_score = ev.rmse(all_true, all_pred)
    mae_score = ev.mae(all_true, all_pred)

    print(f"0 Hidden Layers Metrics:")
    print(f"  MSE: {mse_score:.4f}")
    print(f"  RMSE: {rmse_score:.4f}")
    print(f"  MAE: {mae_score:.4f}")

    # 1 hidden layer
    print("\nRunning 1 hidden layer...")
    results = kfxc.tenfoldcrossvalidationR(X_fires, y_fires,
                                           hidden_layers=1,
                                           nodes_per_hidden_layer=25,
                                           batch_size=124,
                                           learning_rate=1.228307,
                                           momentum=0.062391)

    all_true = np.concatenate([r[0] for r in results])
    all_pred = np.concatenate([r[1] for r in results])
    mse_score = ev.mse(all_true, all_pred)
    rmse_score = ev.rmse(all_true, all_pred)
    mae_score = ev.mae(all_true, all_pred)

    print(f"1 Hidden Layer Metrics:")
    print(f"  MSE: {mse_score:.4f}")
    print(f"  RMSE: {rmse_score:.4f}")
    print(f"  MAE: {mae_score:.4f}")

    # 2 hidden layers
    print("\nRunning 2 hidden layers...")
    results = kfxc.tenfoldcrossvalidationR(X_fires, y_fires,
                                           hidden_layers=2,
                                           nodes_per_hidden_layer=34,
                                           batch_size=88,
                                           learning_rate=0.953211,
                                           momentum=0.150291)

    all_true = np.concatenate([r[0] for r in results])
    all_pred = np.concatenate([r[1] for r in results])
    mse_score = ev.mse(all_true, all_pred)
    rmse_score = ev.rmse(all_true, all_pred)
    mae_score = ev.mae(all_true, all_pred)

    print(f"2 Hidden Layers Metrics:")
    print(f"  MSE: {mse_score:.4f}")
    print(f"  RMSE: {rmse_score:.4f}")
    print(f"  MAE: {mae_score:.4f}")


if __name__ == "__main__":
    demo()
