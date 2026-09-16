import numpy as np


def precision(y_true, y_pred):
    
    true_positives = np.sum((y_true ==1) & (y_pred==1))
    
    predicted_positives = np.sum(y_pred==1)
    
    if predicted_positives == 0:
        return 0.0
    
    return true_positives / predicted_positives

def main():
    
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    
    prec = precision(y_true, y_pred)
    
    print(f"Precision: {prec:.1f}")
    
if __name__ == "__main__":
    main()
