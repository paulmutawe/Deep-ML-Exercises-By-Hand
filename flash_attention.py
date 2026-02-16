import numpy as np

def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int = 2) -> np.ndarray:
    
    sequence_length, model_dimension = Q.shape
    scale = 1.0 / np.sqrt(model_dimension)
    output = np.empty_like(Q, dtype=np.float64)
    
    for query_start in range(0, sequence_length, block_size):
        query_end = min(query_start + block_size, sequence_length)
        query_block = Q[query_start:query_end].astype(np.float64, copy = False)
        
        running_maximum = np.full(query_end - query_start, -np.inf, dtype=np.float64)
        running_sumexp = np.zeros(query_end - query_start, dtype=np.float64)
        output_accumulator = np.zeros((query_end - query_start, model_dimension), dtype=np.float64)
        
        for key_start in range(0, sequence_length, block_size):
            key_end = min(key_start + block_size, sequence_length)
            key_block = K[key_start:key_end].astype(np.float64, copy = False)
            value_block = V[key_start:key_end].astype(np.float64, copy = False)
            
            score_block = (query_block @ key_block.T) * scale
            block_row_max = score_block.max(axis=1)
            
            new_running_maximum = np.maximum(running_maximum, block_row_max)
            old_rescale = np.exp(running_maximum - new_running_maximum)
            score_rescale = np.exp(score_block - new_running_maximum[:, None])
            
            new_running_sumexp = old_rescale * running_sumexp + score_rescale.sum(axis=1)
            output_accumulator = old_rescale[:, None] * output_accumulator + score_rescale @ value_block
            
            running_maximum, running_sumexp = new_running_maximum, new_running_sumexp
            
        output[query_start:query_end] = output_accumulator / running_sumexp[:, None]
    
    return output.astype(Q.dtype, copy=False)

if __name__ == "__main__":
    Q = np.array([[1, 0],
                       [0, 1]], dtype=np.float64)
    K = np.array([[1, 0],
                       [0, 1]], dtype=np.float64)
    V = np.array([[1, 0],
                       [0, 1]], dtype=np.float64)
    
    result = flash_attention_forward(Q, K, V, block_size=1)
    np.set_printoptions(precision=4, suppress=True)
    print(result)
        
