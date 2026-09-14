import numpy as np

def k_split_cross_validation(
    n_samples: int,
    k:int = 5
):
    
    sample_indices = np.arange(n_samples)
    
    data_groups = np.array_split(sample_indices, k)
    
    all_train_validation_splits = []
    
    for validation_group_number in range(k):
        
        validation_indices = data_groups[validation_group_number].tolist()
        
        training_indices = []
        
        for group_number in range(k):
            
            if group_number == validation_group_number:
                continue
            
            training_indices.extend(
                data_groups[group_number].tolist()
            )
            
        all_train_validation_splits.append(
            (training_indices, validation_indices)
        )
    
    return all_train_validation_splits

def main():
    
    splits = k_split_cross_validation(
        n_samples = 10,
        k=5
    )
    
    for split_number in range(len(splits)):
        
        current_split = splits[split_number]
        
        training_indices = current_split[0]
        
        validation_indices = current_split[1]
        
        print(f"Split {split_number + 1}:")
        print(f"Training indices: {training_indices}")
        print(f"Validation indices: {validation_indices}")
        
if __name__ == "__main__":
    main()
