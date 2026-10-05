def balance_undersample(data: list) -> list:
    
    if len(data) ==0:
        return []
    
    labels = []
    
    for sample, label in data:
        labels.append(label)
        
    unique_labels = set(labels)
    
    class_counts = []
    
    for label in unique_labels:
        count = labels.count(label)
        class_counts.append(count)
        
    min_count = min(class_counts)
    
    kept_counts = {}
    
    for label in unique_labels:
        kept_counts[label] = 0
        
    balanced_data = []
    
    for sample, label in data:
        if kept_counts[label] < min_count:
            balanced_data.append((sample, label))
            kept_counts[label] +=1
            
    return balanced_data
    
