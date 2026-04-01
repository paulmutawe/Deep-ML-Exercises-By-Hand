import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray):
    d_k = Q.shape[1]
    
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    masked_scores = scores + mask
    
    exp_scores = np.exp(masked_scores - np.max(masked_scores, axis=1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    output = np.dot(attention_weights, V)
    return output


X = np.array([
	[1, 0, 1, 0],
	[0, 2, 0, 2],
	[1, 1, 1, 1]
], dtype=float)

W_q = np.array([
	[1, 0],
	[0, 1],
	[1, 0],
	[0, 1]
], dtype=float)

W_k = np.array([
	[1, 1],
	[1, 0],
	[0, 1],
	[1, 0]
], dtype=float)

W_v = np.array([
	[1, 0],
	[0, 1],
	[1, 1],
	[0, 0]
], dtype=float)

Q, K, V = compute_qkv(X, W_q, W_k, W_v)

seq_len = Q.shape[0]
mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

print(masked_attention(Q, K, V, mask))
