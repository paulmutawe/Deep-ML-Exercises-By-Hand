import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    
    standardised_data = (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)
    
    normalized_data = (data - np.min(data, axis = 0)) / (np.max(data, axis = 0) - np.min(data, axis = 0))
    
    return np.round(standardised_data, 4), np.round(normalized_data, 4)
    
