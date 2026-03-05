import numpy as np

def lora_forward(
    x: list[list[float]],
    W: list[list[float]],
    A: list[list[float]],
    B: list[list[float]],
    alpha: float = 1.0
) -> list[list[float]]:
    
    x_np = np.asarray(x, dtype=np.float32)
    W_np = np.asarray(W, dtype=np.float32)
    A_np = np.asarray(A, dtype=np.float32)
    B_np = np.asarray(B, dtype=np.float32)
    
    r = A_np.shape[0]
    
    if B_np.shape[1] != r:
        raise ValueError(f"Incompatible shapes: B has rank {B_np.shape[1]} but A has rank {r}.")
    if x_np.shape[1] != W_np.shape[0]:
        raise ValueError(f"Incompatible shapes: x in_features {x_np.shape[1]} != W in_features {W_np.shape[0]}.")
    if W_np.shape[1] != A_np.shape[0]:
        raise ValueError(f"incompatible shapes: W out_features {W_np.shape[1]} != A out_features {A_np.shape[1]}.")
    if B_np.shape[0] != W_np.shape[0]:
        raise ValueError(f"Incompatible shapes: B in_features {B_np.shape[0]} != W in_features {W_np.shape[0]}.")
    
    base = x_np @ W_np
    
    lora_update = (x_np @ B_np) @ A_np
    
    scaling = alpha / r
    
    y = base + scaling * lora_update
    
    return y.tolist()


if __name__ == "__main__":
    x = [[1.0, 2.0], [3.0, 4.0]]
    W = [[0.5, 0.5], [0.5, 0.5]]
    A = [[1.0, 0.0], [0.0, 1.0]]
    B = [[1.0, 1.0], [1.0, 1.0]]
    alpha = 1.0
    
    result = lora_forward(x, W, A, B, alpha)
    print("Output:")
    for row in result:
        print(row)
