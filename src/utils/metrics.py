import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =========================
# Regression metrics (AI)
# =========================

def masked_mae(y_pred, y_true, mask):
    """
    y_pred, y_true, mask: [N, T] or flattened
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    mask   = np.asarray(mask)

    num = np.sum(np.abs(y_pred - y_true) * mask)
    den = np.sum(mask) + 1e-8
    return num / den


def masked_rmse(y_pred, y_true, mask):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    mask   = np.asarray(mask)

    num = np.sum((y_pred - y_true) ** 2 * mask)
    den = np.sum(mask) + 1e-8
    return np.sqrt(num / den)


def masked_corr(y_pred, y_true, mask):
    """
    Masked Pearson correlation
    """
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    mask   = np.asarray(mask).reshape(-1)

    keep = mask > 0.5
    if keep.sum() < 2:
        return np.nan

    yp = y_pred[keep]
    yt = y_true[keep]

    yp = yp - yp.mean()
    yt = yt - yt.mean()

    num = np.sum(yp * yt)
    den = np.sqrt(np.sum(yp**2) * np.sum(yt**2)) + 1e-8
    return num / den


# =========================
# Facies metrics
# =========================

def facies_metrics(y_true, y_pred, labels=None):
    """
    y_true, y_pred: 1D arrays
    """
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    cm  = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "macro_f1": mf1,
        "confusion_matrix": cm
    }