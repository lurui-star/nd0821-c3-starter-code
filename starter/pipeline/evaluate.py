from sklearn.metrics import roc_curve,accuracy_score,f1_score,fbeta_score,recall_score,precision_score, auc

def compute_model_metrics(y, preds,pred_prob):
    """
    Validates the trained machine learning model using precision, recall, F1, and ROC AUC.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels or predicted probabilities (binarized).

    Returns
    -------
    precision : float
        The precision score of the model.
    recall : float
        The recall score of the model.
    fbeta : float
        The F-beta score (F1 score when beta=1).
    fpr : np.array
        False positive rate values for ROC curve.
    tpr : np.array
        True positive rate values for ROC curve.
    roc_auc : float
        Area under the ROC curve.
    """
    pred_prob = pred_prob[:, 1]
    # Calculate precision, recall, and F-beta score using class labels
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    
    # Calculate ROC metrics using predicted probabilities
    fpr, tpr, thresholds = roc_curve(y, pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Optionally print the metrics for display
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-beta (beta=1) score: {fbeta:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    # Return the metrics as a tuple
    return precision, recall, fbeta, fpr, tpr, roc_auc
