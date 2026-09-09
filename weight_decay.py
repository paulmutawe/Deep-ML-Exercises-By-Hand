def apply_weight_decay(
    parameters,
    gradients,
    lr,
    weight_decay,
    apply_to_all
):
    
    updated_parameters = []
    
    for parameter_group_number in range(len(parameters)):
        
        parameter_group = parameters[parameter_group_number]
        gradient_group = gradients[parameter_group_number]
        should_apply_weight_decay = apply_to_all[parameter_group_number]
        
        updated_parameter_group = []
        
        for parameter_number in range(len(parameter_group)):
            
            parameter =parameter_group[parameter_number]
            gradient = gradient_group[parameter_number]
            
            gradient_update = lr * gradient
            
            if should_apply_weight_decay:
                weight_decay_update = lr * weight_decay * parameter
            else:
                weight_decay_update = 0
                
            updated_parameter = (
                parameter 
                - gradient_update
                - weight_decay_update
            )
            
            updated_parameter_group.append(updated_parameter)
            
        updated_parameters.append(updated_parameter_group)
        
    return updated_parameters

def main():
    parameters = [
        [1.0, 2.0],
        [0.5]
    ]

    gradients = [
        [0.1, 0.2],
        [0.05]
    ]

    apply_to_all = [
        True,
        False
    ]

    learning_rate = 0.1
    weight_decay = 0.01

    result = apply_weight_decay(
        parameters,
        gradients,
        learning_rate,
        weight_decay,
        apply_to_all
    )

    print(result)


if __name__ == "__main__":
    main()
                
