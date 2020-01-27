import numpy as np

#obtained from https://github.com/benhamner/Metrics
#implemented by Ben Hamner, CTO of Kaggle

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual: #and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0) #num_hits / (i+1.0) is the precision at position i

    if len(actual) <= 0:
        print "########## WARNING: GROUNDTRUTH LIST IS EMPTY ########## "
        return 0.0

    return score / k #min(len(actual), k)

def neighbor_precision(actual, predicted):

    num_hits = 0.0
    
    for i,p in enumerate(predicted):
        if p in actual: #and p not in predicted[:i]:
            num_hits += 1.0

    prec=1.0
    rec=0.0
    if len(predicted) > 0:
        prec=num_hits/float(len(predicted))
    if len(actual) > 0:
        rec=num_hits/float(len(actual))

    return prec,rec

def label_precision(actual, predicted, label_count):

    num_hits = 0.0
    desired = 0.0

    for label in actual:
        desired = desired + label_count[label]

    for i,p in enumerate(predicted):
        if p in actual: #and p not in predicted[:i]:
            num_hits += 1.0
    
    prec=1.0
    rec=0.0

    if len(predicted) > 0:
        prec = num_hits/float(len(predicted))
    if float(desired) > 0:
        rec = num_hits/float(desired)

    return prec, rec



def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

