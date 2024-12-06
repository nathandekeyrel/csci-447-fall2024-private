import numpy as np
from preprocess import preprocess_data
import tenfoldcv as kfxc
import evaluating as ev


def main():
    ###################################
    # Load datasets
    ###################################
    print("Loading data...")
    cancer_filepath = "../data/breast-cancer-wisconsin.data"
    glass_filepath = "../data/glass.data"
    soybean_filepath = "../data/soybean-small.data"
    abalone_filepath = "../data/abalone.data"
    hardware_filepath = "../data/machine.data"
    fires_filepath = "../data/forestfires.csv"

    ###################################
    # Preprocess Data
    ###################################
    print("Preprocessing datasets...")
    X_cancer, y_cancer = preprocess_data(cancer_filepath)
    X_glass, y_glass = preprocess_data(glass_filepath)
    X_soybean, y_soybean = preprocess_data(soybean_filepath)
    X_abalone, y_abalone = preprocess_data(abalone_filepath)
    X_hardware, y_hardware = preprocess_data(hardware_filepath)
    X_fires, y_fires = preprocess_data(fires_filepath)

    ###################################
    # Run Cancer
    ###################################
    print("=" * 75)
    print("Cancer:")
    print("=" * 75)

    # 0 hidden layers
    print("Running GA...")
    results = kfxc.tenfoldcrossvalidation(X_cancer, y_cancer,
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
    # Run Glass
    ###################################
    print("=" * 75)
    print("\nGlass:")
    print("=" * 75)

    # 0 hidden layers
    print("Running 0 hidden layers...")
    results = kfxc.tenfoldcrossvalidationC(X_glass, y_glass,
                                           hidden_layers=0,
                                           nodes_per_hidden_layer=7,
                                           batch_size=8,
                                           learning_rate=0.3,
                                           momentum=0.95)

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
    results = kfxc.tenfoldcrossvalidationC(X_glass, y_glass,
                                           hidden_layers=1,
                                           nodes_per_hidden_layer=7,
                                           batch_size=16,
                                           learning_rate=0.9,
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
    results = kfxc.tenfoldcrossvalidationC(X_glass, y_glass,
                                           hidden_layers=2,
                                           nodes_per_hidden_layer=7,
                                           batch_size=8,
                                           learning_rate=0.7,
                                           momentum=0.8)

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
    # Run Soybean
    ###################################
    print("=" * 75)
    print("\nSoybean:")
    print("=" * 75)

    # 0 hidden layers
    print("Running 0 hidden layers...")
    results = kfxc.tenfoldcrossvalidationC(X_soybean, y_soybean,
                                           hidden_layers=0,
                                           nodes_per_hidden_layer=38,
                                           batch_size=4,
                                           learning_rate=0.1,
                                           momentum=0.7)

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
    results = kfxc.tenfoldcrossvalidationC(X_soybean, y_soybean,
                                           hidden_layers=1,
                                           nodes_per_hidden_layer=38,
                                           batch_size=4,
                                           learning_rate=0.1,
                                           momentum=0.8)

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
    results = kfxc.tenfoldcrossvalidationC(X_soybean, y_soybean,
                                           hidden_layers=2,
                                           nodes_per_hidden_layer=38,
                                           batch_size=4,
                                           learning_rate=0.5,
                                           momentum=0.7)

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
    # Run Abalone
    ###################################
    print("=" * 75)
    print("\nAbalone:")
    print("=" * 75)

    # 0 hidden layers
    print("Running 0 hidden layers...")
    results = kfxc.tenfoldcrossvalidationR(X_abalone, y_abalone,
                                           hidden_layers=0,
                                           nodes_per_hidden_layer=13,
                                           batch_size=3296,
                                           learning_rate=0.441659,
                                           momentum=0.353981)

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
    results = kfxc.tenfoldcrossvalidationR(X_abalone, y_abalone,
                                           hidden_layers=1,
                                           nodes_per_hidden_layer=13,
                                           batch_size=3247,
                                           learning_rate=0.236368,
                                           momentum=0.367304)

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
    results = kfxc.tenfoldcrossvalidationR(X_abalone, y_abalone,
                                           hidden_layers=2,
                                           nodes_per_hidden_layer=12,
                                           batch_size=2550,
                                           learning_rate=0.158300,
                                           momentum=0.294264)

    all_true = np.concatenate([r[0] for r in results])
    all_pred = np.concatenate([r[1] for r in results])
    mse_score = ev.mse(all_true, all_pred)
    rmse_score = ev.rmse(all_true, all_pred)
    mae_score = ev.mae(all_true, all_pred)

    print(f"2 Hidden Layers Metrics:")
    print(f"  MSE: {mse_score:.4f}")
    print(f"  RMSE: {rmse_score:.4f}")
    print(f"  MAE: {mae_score:.4f}")

    ###################################
    # Running Hardware
    ###################################
    print("=" * 75)
    print("\nHardware:")
    print("=" * 75)

    # 0 hidden layers
    print("Running 0 hidden layers...")
    results = kfxc.tenfoldcrossvalidationR(X_hardware, y_hardware,
                                           hidden_layers=0,
                                           nodes_per_hidden_layer=30,
                                           batch_size=166,
                                           learning_rate=0.011315,
                                           momentum=0.373050)

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
    results = kfxc.tenfoldcrossvalidationR(X_hardware, y_hardware,
                                           hidden_layers=1,
                                           nodes_per_hidden_layer=28,
                                           batch_size=81,
                                           learning_rate=0.012648,
                                           momentum=0.459171)

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
    results = kfxc.tenfoldcrossvalidationR(X_hardware, y_hardware,
                                           hidden_layers=2,
                                           nodes_per_hidden_layer=64,
                                           batch_size=166,
                                           learning_rate=0.005615,
                                           momentum=0.221782)

    all_true = np.concatenate([r[0] for r in results])
    all_pred = np.concatenate([r[1] for r in results])
    mse_score = ev.mse(all_true, all_pred)
    rmse_score = ev.rmse(all_true, all_pred)
    mae_score = ev.mae(all_true, all_pred)

    print(f"2 Hidden Layers Metrics:")
    print(f"  MSE: {mse_score:.4f}")
    print(f"  RMSE: {rmse_score:.4f}")
    print(f"  MAE: {mae_score:.4f}")

    ###################################
    # Running Fires
    ###################################
    print("=" * 75)
    print("\nFires:")
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
    main()
