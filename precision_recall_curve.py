def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    thresholds = sorted(set(y_scores), reverse=True)

    precisions = []
    recalls = []
    actual_positives = y_true.count(1)

    for threshold in thresholds:
        predictions = [1 if score >= threshold else 0 for score in y_scores]
        predicted_positives = predictions.count(1)

        true_positives = sum(
            1 for i in range(len(y_true))
            if predictions[i] == 1 and y_true[i] == 1
        )

        # Edge case: no predicted positives → precision = 1.0
        if predicted_positives == 0:
            precision = 1.0
        else:
            precision = true_positives / predicted_positives

        # Edge case: no actual positives → recall = 0.0
        if actual_positives == 0:
            recall = 0.0
        else:
            recall = true_positives / actual_positives

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls, thresholds

