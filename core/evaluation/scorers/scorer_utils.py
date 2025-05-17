from typing import List


def compute_prf(tp: float, fp: float, fn: float, beta: float = 0.5) -> float:
    """Compute precision, recall, and F-measure given true positives, false positives,
    and false negatives.

    Precision (p) is computed as tp / (tp + fp).
    Recall (r) is computed as tp / (tp + fn).
    F-measure (f) is a weighted harmonic mean of precision and recall.

    Args:
        tp (float): Number of true positives.
        fp (float): Number of false positives.
        fn (float): Number of false negatives.
        beta (float, optional): The weight of recall in the harmonic mean. Defaults to 0.5.

    Returns:
        Tuple[float, float, float]: A tuple containing rounded precision, recall, and F-measure.
    """
    # Calculate precision with division safety check.
    p = float(tp) / (tp + fp) if (tp + fp) else 1.0
    # Calculate recall with division safety check.
    r = float(tp) / (tp + fn) if (tp + fn) else 1.0
    # Calculate F-measure; ensure that denominator is not zero.
    f = float((1 + (beta**2)) * p * r) / (((beta**2) * p) + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def compute_acc(tp: float, fp: float, fn: float, tn: float) -> float:
    """Compute accuracy given true positives, false positives, false negatives, and true negatives.

    Accuracy (acc) is calculated as (tp + tn) / (tp + fp + fn + tn).

    Args:
        tp (float): Number of true positives.
        fp (float): Number of false positives.
        fn (float): Number of false negatives.
        tn (float): Number of true negatives.

    Returns:
        float: The rounded accuracy value.
    """
    total = tp + fp + fn + tn
    acc = float(tp + tn) / total if total else 0.0
    return round(acc, 4)


def gt_numbers(nums_1: List[float], nums_2: List[float]) -> bool:
    """Compare two lists of numbers element-wise.

    Args:
        nums_1 (List[float]): The first list of numbers.
        nums_2 (List[float]): The second list of numbers.

    Raises:
        ValueError: If the lengths of the two lists are not equal.

    Returns:
        bool: True if nums_1 is greater than nums_2 based on the comparison logic, otherwise False.
    """
    if len(nums_1) != len(nums_2):
        raise ValueError("Unequal length of two lists")
    # Compare each corresponding number from both lists.
    for num1, num2 in zip(nums_1, nums_2):
        if num1 > num2:
            return True
        elif num1 < num2:
            return False
    return False
