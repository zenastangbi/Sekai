"""Jaccard and boundary scores for binary masks."""


def jf(target, pred, metrics):
    """Return J, F, and their mean for target and pred boolean masks.

    Use metrics for nonempty masks; score double-empty as one and
    single-empty as zero.
    """
    if not target.any() or not pred.any():
        j = f = float(not target.any() and not pred.any())
    else:
        j = float(metrics.db_eval_iou(target, pred))
        f = float(metrics.db_eval_boundary(target, pred))
    return j, f, (j + f) / 2
