import numpy as np

def train(X_train, y_train, X_val, y_val):
    
    training_feature_means = X_train.mean(axis=0)
    training_target_mean = y_train.mean()
    
    centered_X_train = X_train - training_feature_means
    centered_X_val = X_val - training_feature_means
    
    centered_y_train = y_train - training_target_mean
    
    alpha_values = np.logspace(-2, 4, 13)
    
    lowest_validation_error = np.inf
    best_weights = None
    
    for alpha in alpha_values:
        
        number_of_features = centered_X_train.shape[1]
        
        identity_matrix = np.eye(number_of_features)
        
        ridge_weights = np.linalg.solve(
            centered_X_train.T @ centered_X_train
            + alpha * identity_matrix,
            
            centered_X_train.T @ centered_y_train
        )
        
        validation_predictions = (
            centered_X_val @ ridge_weights
            + training_target_mean
        )
        
        validation_error = np.mean(
            (validation_predictions - y_val) ** 2
        )
        
        if validation_error < lowest_validation_error:
            
            lowest_validation_error = validation_error
            best_weights = ridge_weights 
            
    def predict(X):
        
        centered_X = X - training_feature_means
        
        predictions = (
            centered_X @ best_weights
            + training_target_mean
        )
        
        return predictions

    return predict

if __name__ == "__main__":
   
    X_train = np.array([[1, 1], [2, 1], [3, 2], [4, 3]], dtype=float)
    y_train = np.array([5, 7, 12, 17], dtype=float)

    X_val = np.array([[2, 2], [3, 3]], dtype=float)
    y_val = np.array([10, 15], dtype=float)

    predict = train(X_train, y_train, X_val, y_val)

    X_test = np.array([[1, 2], [5, 4]], dtype=float)
    print("Predictions:", predict(X_test))
