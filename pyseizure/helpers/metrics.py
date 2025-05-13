import numpy as np
from torch import Tensor
import torchmetrics
from sklearn.metrics import accuracy_score, precision_recall_curve, \
    roc_curve, auc, PrecisionRecallDisplay, f1_score, precision_score, \
    mean_squared_error, confusion_matrix, recall_score


def get_accuracy(output, target):
    """Obtain accuracy for training round."""
    return accuracy_score(target, output.round())


def get_roc(output, target):
    fpr, tpr, _ = roc_curve(target, output, pos_label=1)

    return auc(fpr, tpr)


def get_prauc(output, target, ax=None):
    precision, recall, _ = precision_recall_curve(target, output,
                                                  pos_label=1)
    if ax is not None:
        PrecisionRecallDisplay(precision=precision,
                               recall=recall).plot(ax=ax)

    return auc(recall, precision)


def get_f1(output, target):
    return f1_score(target, output.round(), average='macro', pos_label=1)


def get_precision(output, target):
    return precision_score(target, output.round(), average='macro',
                           pos_label=1, zero_division=0.0)


def get_mse(output, target):
    return mean_squared_error(target, output)


def get_sensitivity(output, target):
    """
    Sensitivity / True Positive Rate / Recall   = TP / (TP + FN)
    False Negative Rate                         = FN / (TP + FN)
    False Positive Rate                         = FP / (TN + FP)
    Specificity / True Negative Rate            = TN / (TN + FP)
    """
    # _, _, fn, tp = get_confusion_matrix(output.round(), target).ravel()
    # try:
    #     sensitivity = tp / (tp + fn)
    # except ZeroDivisionError:
    #     sensitivity = 0.0
    sensitivity_tm = torchmetrics.Recall(task='multiclass', num_classes=2)
    if len(output.shape) == 2:
        sensitivity_tm.update(Tensor(output), Tensor(target))
    else:
        sensitivity_tm.update(Tensor(np.concatenate(
            [(1 - output).reshape(-1, 1), output.reshape(-1, 1)], axis=1)),
                              Tensor(target))

    return sensitivity_tm.compute()


def get_recall(output, target):
    return recall_score(target, output.round(), average="macro")


def get_specificity(output, target):
    # tn, fp, _, _ = get_confusion_matrix(output.round(), target).ravel()
    # try:
    #     specificity = tn / (tn + fp)
    # except ZeroDivisionError:
    #     specificity = 0.0
    specificity_tm = torchmetrics.Specificity(task='multiclass', num_classes=2,
                                              average='macro')
    if len(output.shape) == 2:
        specificity_tm.update(Tensor(output), Tensor(target))
    else:
        specificity_tm.update(Tensor(np.concatenate(
            [(1 - output).reshape(-1, 1), output.reshape(-1, 1)], axis=1)),
                              Tensor(target))

    return specificity_tm.compute()


def get_confusion_matrix(output, target):
    """
    Get confusion matrix.

    Parameters
    ----------
    output
    target

    Returns
    -------
    Tuple
        tn, fp, fn, tp
    """
    if all(target == 0) and all(output == 0):
        return np.array([len(target), 0, 0, 0])
    elif all(target == 1) and all(output == 1):
        return np.array([0, 0, 0, len(target)])

    return confusion_matrix(target, output).ravel()


class PostProcessor:
    def __init__(self,
                 y_true: np.array,
                 y_pred: np.array,
                 min_len: int = 1,
                 max_gap: int = 2,
                 frequency: int = 256,
                 epoch: int = 256):
        self.y_true = y_true
        self.y_pred = y_pred
        self.epoch_len = epoch / frequency
        self.min_len = min_len
        self.max_gap = max_gap

    @staticmethod
    def _get_event_segments(array: np.array):
        """Extract (start, end) indices of events from a binary sequence."""
        diff = np.diff(np.pad(array, (1, 1), constant_values=0))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        if starts.size > 0:
            return np.column_stack((starts, ends))

        return np.array([]).reshape(0, 2)

    def ovlp(self):
        """Compute OVLP (any-overlap) metric for binary sequences."""
        tn, fp, fn, tp = 0, 0, 0, 0

        true_events = self._get_event_segments(self.y_true)
        pred_events = self._get_event_segments(self.y_pred)

        matched_gt = np.zeros(len(true_events), dtype=bool)

        for pred_start, pred_end in pred_events:
            overlap = ((true_events[:, 1] >= pred_start)
                       & (true_events[:, 0] <= pred_end))
            if np.any(overlap):
                tp += np.count_nonzero(overlap)
                matched_gt[overlap] = True

        fp = len(pred_events) - tp
        fp = fp if fp > 0 else 0
        fn = len(true_events) - np.sum(matched_gt)

        if fp + fn + tp == 0:
            tn += 1

        return tn, fp, fn, tp

    def get_score(self):
        ovlp = self.ovlp()
        ovlp_sensitivity = ovlp[3] / (ovlp[3] - ovlp[2])

        _, fp, _, _ = get_confusion_matrix(self.y_true, self.y_pred)
        fa = fp / (len(self.y_true) * self.epoch_len / 3600)

        return (100 * ovlp_sensitivity) - (0.4 * fa)

    def process(self,
                tolerance: int = 1,
                array: np.array = None,
                threshold: float = None):
        if array is None:
            array = self.y_pred
        array = array.copy()

        if threshold is not None:
            events = self._get_event_segments(
                self._threshold_predictions(array, threshold))
        else:
            events = self._get_event_segments(array)
        for start, end in events:
            if (end - start < tolerance and end - tolerance >= 0
                    and end + 1 != len(array)):
                array[start:end + 1] = 0

        gaps = self._get_event_segments(1 - array)
        for start, end in gaps:
            if (end - start < tolerance and end - tolerance >= 0
                    and end + 1 != len(array)):
                array[start:end + 1] = 1

        return array

    @staticmethod
    def _threshold_predictions(y_pred, threshold=0.5):
        """
        Convert probability predictions into binary labels using a threshold.
        """
        return (y_pred >= threshold).astype(int)

    def metrics_with_drift(self,
                           threshold: float = 0.5,
                           tolerance: int = 1):
        """
        Computes multiple drift-aware metrics: F1-score, accuracy, precision,
        recall, ROC AUC, specificity, PRAUC, and MSE.

        Parameters:
        threshold: float
            probability threshold.
        tolerance: int
            maximum drift allowed for an event in epochs.

        Returns:
        Dict
            dictionary with computed metrics.
        """
        gt_events = self._get_event_segments(self.y_true)
        gt_n_events = self._get_event_segments(1 - self.y_true)
        pred_events = self._get_event_segments(
            self._threshold_predictions(self.y_pred, threshold))
        pred_n_events = self._get_event_segments(
            1 - self._threshold_predictions(self.y_pred, threshold))

        y_pred = self.y_pred.copy()
        y_true = self.y_true.copy()
        if tolerance > 0:
            y_pred = self.process(tolerance, self.y_pred,
                                  threshold).astype(float)
            y_true = self.process(tolerance, self.y_true)

            gt_events = self._get_event_segments(y_true)
            gt_n_events = self._get_event_segments(1 - y_true)
            pred_events = self._get_event_segments(y_pred)
            pred_n_events = self._get_event_segments(1 - y_pred)

        event_lists = [gt_events, pred_events, gt_n_events, pred_n_events]

        fnfp_indices = np.where(np.logical_xor(
            y_true, self._threshold_predictions(y_pred)))[0]
        fnfp_events = set(fnfp_indices[np.concatenate([
            np.where(np.diff(fnfp_indices) == 1)[0],
            np.where(np.diff(fnfp_indices) == 1)[0] + 1])])

        valid_indices = set()
        for events in event_lists:
            for start, end in events:
                valid_indices.update(range(start, end + 1))
        tbr = list(set(fnfp_indices) & valid_indices)

        tbr = np.array([x for x in tbr if x not in fnfp_events])

        if len(tbr) > 0:
            y_pred[tbr[(y_pred[tbr] > threshold)
                       & (y_true[tbr] == 0)]] = threshold - 0.01
            y_pred[tbr[(y_pred[tbr] <= threshold)
                       & (y_true[tbr] == 1)]] = threshold + 0.01

        mse = get_mse(y_pred, y_true)
        roc_auc = get_roc(y_pred, y_true)
        accuracy = get_accuracy(y_pred, y_true)
        precision = get_precision(y_pred, y_true)
        f1 = get_f1(y_pred, y_true)
        pr_auc = get_prauc(y_pred, y_true)
        sensitivity = get_sensitivity(y_pred, y_true)
        specificity = get_specificity(y_pred, y_true)
        tn, fp, fn, tp = get_confusion_matrix(y_pred.round(), y_true)

        return {
            "mse": mse,
            "roc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "f1": f1,
            "prauc": pr_auc,
            "sensitivity": sensitivity.item(),
            "specificity": specificity.item(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }, y_pred
