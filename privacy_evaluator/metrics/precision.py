# Helper function to calculate precision of an ART attack model

def calc_precision(predicted, actual, positive_value=1):
    
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1
    
    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    return precision