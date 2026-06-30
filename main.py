import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import config
from dataset import prepare_data, to_bipolar_one_hot
from model import MLP

def run_experiment(exp_name, method, hidden_logic, activation):
    print(f"\n{'='*50}")
    print(f"------RUNNING EXPERIMENT: {exp_name}")

    X_train, X_test, y_train, y_test = prepare_data(method=method)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train_hot = to_bipolar_one_hot(y_train)

    input_neurons = X_train.shape[1]
    output_neurons = 10

    if hidden_logic == 'single_mean':
        hidden_sizes = [int((input_neurons + output_neurons) / 2)]
    elif hidden_logic == 'double_mean':
        h = int((input_neurons + output_neurons) / 2)
        hidden_sizes = [h,h]
    elif hidden_logic =='two_thirds_half':
        hidden_sizes = [int(input_neurons * 0.66), int(input_neurons * 0.5)]
    elif hidden_logic == 'half_one_third':
        hidden_sizes = [int(input_neurons * 0.5), int(input_neurons * 0.33)]
    else:
        raise ValueError('Unknown hidden logic!')
    
    layer_sizes = [input_neurons] + hidden_sizes + [output_neurons]
    print(f"Architecture: {layer_sizes} | Activation: {activation} | Preprocess: {method}")
    

    model = MLP(layer_sizes, activation=activation, lr=config.LEARNING_RATE)
    mse_history = []

    X_train_gpu = cp.asarray(X_train)
    y_train_hot_gpu = cp.asarray(y_train_hot)
    X_test_gpu = cp.asarray(X_test)

    for epoch in range(config.EPOCHS):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        for idx in indices:
            model.forward(X_train_gpu[idx])
            model.backward(y_train_hot_gpu[idx])

        mse_gpu = model.compute_mse(X_train_gpu, y_train_hot_gpu)
        mse = float(mse_gpu)
        mse_history.append(mse)
        print(f"Epoch {epoch+1:02d}/{config.EPOCHS} | MSE: {mse:.4f}")

    y_pred_gpu = model.predict(X_test_gpu)
    y_pred = cp.asnumpy(y_pred_gpu) 
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n {exp_name} Final Accuracy: {acc * 100:.2f}%")

    plt.figure()
    plt.plot(range(1, config.EPOCHS+1), mse_history, marker='o')
    plt.title(f'{exp_name} - MSE vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    safe_name = exp_name.replace(' ', '_').replace('/', '_')
    plt.savefig(os.path.join(config.OUTPUT_DIR, f"{safe_name}_plot.png"))
    plt.close()
    
    return acc


def main():
    experiments = [
        # Part A: Compare Mean vs Pad
        {"name": "Part A - Mean Pooling", "method": "mean", "hidden": "single_mean", "act": "bipolar_sigmoid"},
        {"name": "Part A - Zero Padding", "method": "pad", "hidden": "single_mean", "act": "bipolar_sigmoid"},
        
        # Part B: Two hidden layers, different sizes, using 'mean' for speed
        {"name": "Part B1 - Double Mean", "method": "mean", "hidden": "double_mean", "act": "bipolar_sigmoid"},
        {"name": "Part B2 - 2/3 and 1/2", "method": "mean", "hidden": "two_thirds_half", "act": "bipolar_sigmoid"},
        {"name": "Part B3 - 1/2 and 1/3", "method": "mean", "hidden": "half_one_third", "act": "bipolar_sigmoid"},
        
        # Part C: ReLU Activation
        {"name": "Part C1 - Double Mean - Relu", "method": "mean", "hidden": "double_mean", "act": "relu"},
        {"name": "Part C2 - 2/3 and 1/2 - Relu", "method": "mean", "hidden": "two_thirds_half", "act": "relu"},
        {"name": "Part C3 - 1/2 and 1/3 - Relu", "method": "mean", "hidden": "half_one_third", "act": "relu"},
    ]
    
    results = {}
    for exp in experiments:
        acc = run_experiment(exp["name"], exp["method"], exp["hidden"], exp["act"])
        results[exp["name"]] = acc
        
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    for name, acc in results.items():
        print(f"{name:<30}: {acc*100:.2f}%")

if __name__ == "__main__":
    main()