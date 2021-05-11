# Helper function to calculate recall of an ART attack model

def calc_precision_recall(predicted, actual, positive_value=1):
    
    score = 0  # both predicted and actual are positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1
    
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return recall