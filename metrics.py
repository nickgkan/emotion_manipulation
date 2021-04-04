"""Metric utilities."""

from sklearn.metrics import average_precision_score


def compute_ap(gt, pred, average=None):
    """
    Compute the multi-label Average Precision.

    Inputs:
        gt (np.ndarray): (n_data, n_labels), binary labels
        pred (np.ndarray): (n_data, n_labels), probabilities
    Returns:
        AP (list): average precision for all classes
    """
    AP = []
    for cid in range(gt.shape[1]):
        gt_cls = gt[:, cid].astype('float32')
        pred_cls = pred[:, cid].astype('float32')
        pred_cls -= 1e-5 * gt_cls
        ap = average_precision_score(gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP
