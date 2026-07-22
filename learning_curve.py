import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    
    X_train = np.asarray(X_train, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    X_val = np.asarray(X_val, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    
    def polynomial_features(x):
        
        return np.vander(x, N=degree + 1, increasing = True)
    
    X_val_poly = polynomial_features(X_val)
    
    train_errors = []
    val_errors = []
    
    for n in train_sizes:
        
        X_subset = X_train[:n]
        y_subset = y_train[:n]
        
        X_subset_poly = polynomial_features(X_subset)
        
        weights = np.linalg.pinv(X_subset_poly) @ y_subset
        
        train_predictions = X_subset_poly @ weights
        val_predictions = X_val_poly @ weights 
        
        train_mse = np.mean((train_predictions - y_subset) ** 2)
        val_mse = np.mean((val_predictions - y_val) ** 2)
        
        train_errors.append(float(train_mse))
        val_errors.append(float(val_mse))
        
    final_train_error = train_errors[-1]
    final_val_error = val_errors[-1]
    
    if final_train_error > bias_threshold:
        diagnosis = "high_bias"
    elif final_val_error - final_train_error > variance_threshold:
        diagnosis = "high_variance"
    else:
        diagnosis = "good_fit"
        
    return {
        "train_errors": train_errors,
        "val_errors": val_errors,
        "diagnosis": diagnosis
    }
